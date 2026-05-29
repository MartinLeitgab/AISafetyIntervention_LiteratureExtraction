"""experiment_xrisk_unmerge_ec.py

Test the hypothesis (gleb_analysis_critique_DO_NOT_COMMIT.md, Claim 3) that
Gleb's "race dynamics dominates top-100 risks by eigenvector centrality"
finding is an artifact of (a) merging xrisk near-duplicates, which (b)
concentrates degree on a small number of merged-hub risk nodes, which (c)
inherit high EC, and these (d) tend to co-occur with race-framing PAs in
source papers.

Method:
  Variant A (baseline, Gleb-equivalent): EC on combined structural+similarity
  graph at SIM>=0.8 (his threshold). Top-100 risk nodes by EC. For each,
  count PA neighbors containing 'race' or 'competitive' (Gleb's race-framing
  definition, broadened to "any PA neighbor" first then "sole PA neighbor").

  Variant B (xrisk hubs removed): same graph minus risk nodes whose name
  contains 'existential' or 'extinction'. Recompute EC, retake top-100,
  recount race-framed.

  Variant C (no similarity edges, all xrisk kept): EDGE-only graph, top-100
  by EC, race-framed count. Sanity check that the artifact is specifically
  similarity-edge-driven, not just any-edge driven.

Class B: no LLM calls. Outputs report to phase2_results/.
"""

from __future__ import annotations
import pickle
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
OUT = ROOT / "phase2_results/experiment_xrisk_unmerge_ec_report.json"

XRISK_KEYWORDS = ["existential", "extinction"]
RACE_KEYWORDS = ["race", "competitive", "competition", "racing"]


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def load_data():
    print("loading node_attrs ...", flush=True)
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    print(f"  {len(na)} nodes", flush=True)
    print("loading edge_data ...", flush=True)
    with open(STEP1 / "graph_edge_data.pkl", "rb") as f:
        ed = pickle.load(f)
    print(f"  {len(ed)} edges", flush=True)
    return na, ed


def build_sparse(
    node_attrs, edges, include_edge=True, include_sim_at=0.8, excluded_nids: set = None
):
    """Build undirected sparse adjacency matrix.
    Returns (csr_matrix, nid_to_idx, idx_to_nid)."""
    excluded_nids = excluded_nids or set()
    nids = sorted(n for n in node_attrs if n not in excluded_nids)
    nid_to_idx = {n: i for i, n in enumerate(nids)}
    rows, cols = [], []
    n_edge_added = 0
    n_sim_added = 0
    n_skipped_excluded = 0
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s is None or t is None:
            continue
        if s == t:
            continue
        if s in excluded_nids or t in excluded_nids:
            n_skipped_excluded += 1
            continue
        etype = (e.get("type") or "").upper()
        if etype == "EDGE":
            if not include_edge:
                continue
            if s in nid_to_idx and t in nid_to_idx:
                rows.append(nid_to_idx[s])
                cols.append(nid_to_idx[t])
                rows.append(nid_to_idx[t])
                cols.append(nid_to_idx[s])
                n_edge_added += 1
        elif etype == "SIMILARITY":
            if include_sim_at is None:
                continue
            score = e.get("similarity_score")
            if score is None:
                continue
            if cos_sim_from_score(score) < include_sim_at:
                continue
            if s in nid_to_idx and t in nid_to_idx:
                rows.append(nid_to_idx[s])
                cols.append(nid_to_idx[t])
                rows.append(nid_to_idx[t])
                cols.append(nid_to_idx[s])
                n_sim_added += 1
    data = np.ones(len(rows), dtype=np.float32)
    n = len(nids)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    print(
        f"  built sparse adjacency: {n} nodes, {n_edge_added} EDGE + "
        f"{n_sim_added} SIM (sim>={include_sim_at}) = "
        f"{A.nnz // 2} undirected unique edges; skipped {n_skipped_excluded} "
        f"edges touching excluded nodes",
        flush=True,
    )
    return A, nid_to_idx, nids


def eigenvector_centrality_sparse(A, max_iter=200, tol=1e-6):
    """Power iteration EC on undirected sparse adjacency. Returns vector."""
    n = A.shape[0]
    x = np.ones(n, dtype=np.float64) / np.sqrt(n)
    for it in range(max_iter):
        y = A @ x
        norm = np.linalg.norm(y)
        if norm == 0:
            break
        y = y / norm
        diff = np.linalg.norm(y - x)
        x = y
        if diff < tol:
            print(
                f"  EC converged in {it + 1} iterations (diff={diff:.2e})", flush=True
            )
            return x
    print(
        f"  EC did NOT converge in {max_iter} iters (final diff={diff:.2e})", flush=True
    )
    return x


def get_pa_neighbors_for_risk(risk_nid, node_attrs, edges_by_endpoint):
    """Return list of PA-node nids that are first-hop (EDGE-type) neighbors of risk_nid."""
    neighbors = edges_by_endpoint.get(risk_nid, set())
    pa_neighbors = []
    for n in neighbors:
        attrs = node_attrs.get(n) or {}
        if (attrs.get("concept_category") or "").lower() == "problem analysis":
            pa_neighbors.append(n)
    return pa_neighbors


def is_race_framed_name(name):
    name_lower = (name or "").lower()
    return any(kw in name_lower for kw in RACE_KEYWORDS)


def race_check(risk_nid, node_attrs, edges_by_endpoint, mode="any_pa_neighbor"):
    """Per Gleb: 'sole PA neighbor contains race/competitive'.
    Our checks:
      'any_pa_neighbor' — any PA neighbor matches (broader)
      'sole_pa_neighbor' — single PA neighbor and it matches (strictest, Gleb's)
    Returns bool.
    """
    pa = get_pa_neighbors_for_risk(risk_nid, node_attrs, edges_by_endpoint)
    if not pa:
        return False
    if mode == "sole_pa_neighbor":
        if len(pa) != 1:
            return False
        return is_race_framed_name(node_attrs.get(pa[0], {}).get("name", ""))
    elif mode == "any_pa_neighbor":
        return any(
            is_race_framed_name(node_attrs.get(n, {}).get("name", "")) for n in pa
        )
    return False


def edges_by_endpoint_edgeonly(edges, excluded_nids=None):
    """Map nid -> set of EDGE-type neighbors (exclude similarity).
    Used for race-framing detection: PA neighbors are always structural, not sim."""
    excluded_nids = excluded_nids or set()
    m = defaultdict(set)
    for e in edges:
        if (e.get("type") or "").upper() != "EDGE":
            continue
        s, t = e.get("source"), e.get("target")
        if s is None or t is None or s == t:
            continue
        if s in excluded_nids or t in excluded_nids:
            continue
        m[s].add(t)
        m[t].add(s)
    return m


def identify_xrisk_hubs(node_attrs):
    """Return set of nids whose name OR description contains an xrisk keyword."""
    out = set()
    for nid, attrs in node_attrs.items():
        if (attrs.get("concept_category") or "").lower() != "risk":
            continue
        name = (attrs.get("name") or "").lower()
        if any(kw in name for kw in XRISK_KEYWORDS):
            out.add(nid)
    return out


def top_100_race_analysis(
    A, nids, nid_to_idx, node_attrs, ec_vector, edges_by_endpoint, label, top_n=100
):
    """Given an EC vector and risk-node filter, return top-N risks by EC and race-framed
    counts under both 'sole_pa_neighbor' and 'any_pa_neighbor' definitions."""
    risk_idx_ec = [
        (nid_to_idx[n], n, ec_vector[nid_to_idx[n]])
        for n in nids
        if (node_attrs.get(n, {}).get("concept_category") or "").lower() == "risk"
    ]
    risk_idx_ec.sort(key=lambda x: -x[2])
    top = risk_idx_ec[:top_n]
    n_race_sole = sum(
        1
        for _, nid, _ in top
        if race_check(nid, node_attrs, edges_by_endpoint, mode="sole_pa_neighbor")
    )
    n_race_any = sum(
        1
        for _, nid, _ in top
        if race_check(nid, node_attrs, edges_by_endpoint, mode="any_pa_neighbor")
    )
    # Top-10 named for quick inspection
    top_10_named = [
        {
            "nid": nid,
            "ec": float(ec_vector[idx]),
            "name": node_attrs.get(nid, {}).get("name", ""),
            "race_sole": race_check(
                nid, node_attrs, edges_by_endpoint, mode="sole_pa_neighbor"
            ),
            "race_any": race_check(
                nid, node_attrs, edges_by_endpoint, mode="any_pa_neighbor"
            ),
            "n_pa_neighbors": len(
                get_pa_neighbors_for_risk(nid, node_attrs, edges_by_endpoint)
            ),
        }
        for idx, nid, _ in top[:10]
    ]
    return {
        "label": label,
        "top_n": top_n,
        "n_total_risks_ranked": len(risk_idx_ec),
        "race_sole_pa_count": n_race_sole,
        "race_sole_pa_pct": round(100 * n_race_sole / top_n, 1),
        "race_any_pa_count": n_race_any,
        "race_any_pa_pct": round(100 * n_race_any / top_n, 1),
        "top_10_named": top_10_named,
    }


def main():
    na, ed = load_data()
    xrisk_hubs = identify_xrisk_hubs(na)
    print(
        f"\nidentified {len(xrisk_hubs)} risk nodes matching xrisk keywords "
        f"({XRISK_KEYWORDS})",
        flush=True,
    )

    # Race-framing detection uses EDGE-type neighbors only (PA-axis, structural)
    print(
        "\nbuilding edges_by_endpoint (EDGE-only, for PA-neighbor lookup) ...",
        flush=True,
    )
    edges_by_endpoint = edges_by_endpoint_edgeonly(ed)
    print(
        f"  {len(edges_by_endpoint)} nodes have at least 1 EDGE-type neighbor",
        flush=True,
    )

    results = {}

    # ===== Variant A: full graph + similarity at 0.8 (Gleb-equivalent) =====
    print("\n" + "=" * 70)
    print("VARIANT A: full graph + SIM>=0.8 (Gleb-equivalent)")
    print("=" * 70)
    A_a, n2i_a, nids_a = build_sparse(na, ed, include_edge=True, include_sim_at=0.8)
    ec_a = eigenvector_centrality_sparse(A_a)
    results["A_full_with_sim"] = top_100_race_analysis(
        A_a, nids_a, n2i_a, na, ec_a, edges_by_endpoint, "A_full_with_sim"
    )
    del A_a, ec_a  # free memory

    # ===== Variant B: xrisk hubs removed + similarity at 0.8 =====
    print("\n" + "=" * 70)
    print("VARIANT B: xrisk hubs REMOVED + SIM>=0.8")
    print("=" * 70)
    A_b, n2i_b, nids_b = build_sparse(
        na, ed, include_edge=True, include_sim_at=0.8, excluded_nids=xrisk_hubs
    )
    ec_b = eigenvector_centrality_sparse(A_b)
    results["B_no_xrisk_with_sim"] = top_100_race_analysis(
        A_b, nids_b, n2i_b, na, ec_b, edges_by_endpoint, "B_no_xrisk_with_sim"
    )
    del A_b, ec_b

    # ===== Variant C: full graph + EDGE-only (no similarity) =====
    print("\n" + "=" * 70)
    print("VARIANT C: full graph + EDGE-only (no similarity edges)")
    print("=" * 70)
    A_c, n2i_c, nids_c = build_sparse(na, ed, include_edge=True, include_sim_at=None)
    ec_c = eigenvector_centrality_sparse(A_c)
    results["C_edge_only"] = top_100_race_analysis(
        A_c, nids_c, n2i_c, na, ec_c, edges_by_endpoint, "C_edge_only"
    )
    del A_c, ec_c

    # ===== Variant D: xrisk REMOVED + EDGE-only =====
    print("\n" + "=" * 70)
    print("VARIANT D: xrisk hubs REMOVED + EDGE-only")
    print("=" * 70)
    A_d, n2i_d, nids_d = build_sparse(
        na, ed, include_edge=True, include_sim_at=None, excluded_nids=xrisk_hubs
    )
    ec_d = eigenvector_centrality_sparse(A_d)
    results["D_no_xrisk_edge_only"] = top_100_race_analysis(
        A_d, nids_d, n2i_d, na, ec_d, edges_by_endpoint, "D_no_xrisk_edge_only"
    )
    del A_d, ec_d

    # ===== Summary table =====
    print("\n" + "=" * 70)
    print("SUMMARY — race-framed fraction in top-100 risks by EC")
    print("=" * 70)
    print(f"{'Variant':<35} {'sole_PA':>10} {'any_PA':>10}")
    for k in [
        "A_full_with_sim",
        "B_no_xrisk_with_sim",
        "C_edge_only",
        "D_no_xrisk_edge_only",
    ]:
        r = results[k]
        print(f"  {k:<33} {r['race_sole_pa_pct']:>9.1f}% {r['race_any_pa_pct']:>9.1f}%")

    out_doc = {
        "experiment": "xrisk-unmerge effect on EC race-framing finding",
        "xrisk_keywords": XRISK_KEYWORDS,
        "race_keywords": RACE_KEYWORDS,
        "n_xrisk_nodes_removed": len(xrisk_hubs),
        "n_total_risk_nodes_in_corpus": sum(
            1
            for a in na.values()
            if (a.get("concept_category") or "").lower() == "risk"
        ),
        "variants": results,
    }
    OUT.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
