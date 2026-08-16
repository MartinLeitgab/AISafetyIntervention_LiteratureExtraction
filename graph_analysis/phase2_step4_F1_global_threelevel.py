"""
phase2_step4_F1_global_threelevel.py

Frozenset rebuild + 3-level Hamming family analysis for global-cutoff
clustering output. Used by phase2_step4_global_cutoff_scan orchestration.

Family-definition levels:
  STRICT_TUPLE: frozenset of (semantic_cid, role_label) tuples
                — same frozenset only if both semantic AND role match
  SEMANTIC_ONLY: frozenset of semantic_cid alone
                — captures cross-role overlap (concept reused in different roles)
  ROLE_PATTERN: tuple of (role_label, count) sorted by role
                — captures chain skeleton (how many of each role) ignoring content

For each level, computes:
  - # frozensets and # paths covered
  - mean / median pairwise Hamming (sym-diff) or Jaccard distance
  - # connected-components families at distance <=2
  - top-10 families by #paths
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4 = ROOT / "phase2_results/step4_finalanalysis"
PATHS_DIR = ROOT / "phase1_rawpathsfiles"
EDGE_ONLY_PATH = PATHS_DIR / "paths_hopwise_v4_edge_only.jsonl"

LOGICAL_ORDER = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]
ROLE_PREFIX = {
    "problem_analysis": "pa",
    "theoretical_insight": "ti",
    "design_rationale": "dr",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
}
BODY_ROLES = set(LOGICAL_ORDER)


def fmt_strict_tuple(elem):
    """elem = (semantic_cid, role_label)"""
    cid, role = elem
    return f"{ROLE_PREFIX.get(role, role[:2])}:{cid}"


def sig_to_str_strict(sig):
    """Order strict-tuple frozenset by logical role, then by cid."""
    by_role = defaultdict(list)
    for cid, role in sig:
        by_role[role].append(str(cid))
    parts = []
    for role in LOGICAL_ORDER:
        if role in by_role:
            for cid in sorted(
                by_role[role], key=lambda x: int(x) if x.lstrip("-").isdigit() else 0
            ):
                parts.append(f"{ROLE_PREFIX[role]}:{cid}")
    return " & ".join(parts)


def sig_to_str_semantic(sig):
    """Order semantic-only frozenset by cid."""
    parts = sorted(
        [f"sem:{cid}" for cid in sig],
        key=lambda x: int(x[4:]) if x[4:].lstrip("-").isdigit() else 0,
    )
    return " & ".join(parts)


def role_pattern_to_str(rp):
    """rp = tuple of (role, count) — render in logical order."""
    d = dict(rp)
    parts = [
        f"{ROLE_PREFIX[r]}={d.get(r, 0)}" for r in LOGICAL_ORDER if d.get(r, 0) > 0
    ]
    return " ".join(parts)


def hamming_families(sig_list, max_dist=2):
    """Connected-components families at sym-diff distance <= max_dist.
    sig_list: list of frozenset (or tuple).
    Returns: list of components (each a list of indices), sorted by size desc.
    """
    n = len(sig_list)
    adj = defaultdict(set)
    for i in range(n):
        si = (
            set(sig_list[i])
            if not isinstance(sig_list[i], (set, frozenset))
            else sig_list[i]
        )
        for j in range(i + 1, n):
            sj = (
                set(sig_list[j])
                if not isinstance(sig_list[j], (set, frozenset))
                else sig_list[j]
            )
            d = len(si ^ sj)
            if d <= max_dist:
                adj[i].add(j)
                adj[j].add(i)
    seen = set()
    comps = []
    for start in range(n):
        if start in seen:
            continue
        stack = [start]
        comp = []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack.extend(adj[v] - seen)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: -len(c))
    return comps


def jaccard_distance_matrix(sig_list):
    """Pairwise Jaccard distance (1 - |A∩B|/|A∪B|)."""
    n = len(sig_list)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        si = set(sig_list[i])
        for j in range(i + 1, n):
            sj = set(sig_list[j])
            u = len(si | sj)
            if u == 0:
                d = 0.0
            else:
                d = 1.0 - len(si & sj) / u
            out[i, j] = d
            out[j, i] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, required=True)
    ap.add_argument("--paths-file", default=str(EDGE_ONLY_PATH))
    ap.add_argument(
        "--max-hamming",
        type=int,
        default=2,
        help="max sym-diff distance for Hamming family connection",
    )
    args = ap.parse_args()
    cutoff = args.cutoff
    suffix = f"global_cutoff{cutoff:.2f}"

    print("=" * 70)
    print(f"F1 + 3-level Hamming on global cluster_memberships, cutoff={cutoff:.2f}")
    print("=" * 70)

    # Load PKL + role map
    pkl_file = STEP1 / f"cluster_memberships_rev8_{suffix}.pkl"
    role_file = STEP1 / f"role_of_rev8_{suffix}.pkl"
    print(f"\nLoading {pkl_file.name} ...")
    with open(pkl_file, "rb") as f:
        cm = pickle.load(f)
    with open(role_file, "rb") as f:
        role_of = pickle.load(f)
    print(f"  {len(cm)} cluster records, {len(role_of)} node roles")

    # Build node -> (group, cid) maps
    node_to_body = {}  # nid -> sem_cid (global body cluster, no role)
    node_to_risk = {}
    node_to_interv = {}
    for key, members in cm.items():
        group = str(key[2])
        cid = str(key[4])
        for nid in members:
            nid_int = int(nid)
            if group == "body_global":
                node_to_body[nid_int] = cid
            elif group == "risk":
                node_to_risk[nid_int] = cid
            elif group == "intervention":
                node_to_interv[nid_int] = cid
    print(f"  body-clustered : {len(node_to_body):,}")
    print(f"  risk-clustered : {len(node_to_risk):,}")
    print(f"  intv-clustered : {len(node_to_interv):,}")

    # Read EDGE-only paths, apply strict filter, build 3 frozenset views
    sigs_strict = Counter()  # frozenset of (sem_cid, role) tuples
    sigs_semantic = Counter()  # frozenset of sem_cid
    sigs_rolepat = Counter()  # tuple of (role, count) sorted

    sig_strict_RI_pairs = defaultdict(set)
    sig_semantic_RI_pairs = defaultdict(set)
    sig_rolepat_RI_pairs = defaultdict(set)

    n_total = n_strict = 0
    print(f"\nReading {Path(args.paths_file).name} ...")
    with open(args.paths_file) as f:
        for line in f:
            obj = json.loads(line)
            seq = obj.get("path") or obj.get("node_id_sequence")
            if not seq or len(seq) < 3:
                continue
            n_total += 1
            path = [int(x) for x in seq]
            r_n, i_n = path[0], path[-1]
            if r_n not in node_to_risk or i_n not in node_to_interv:
                continue
            body = path[1:-1]
            if any(b not in node_to_body for b in body):
                continue
            # All clustered. Build the three signatures.
            n_strict += 1
            # Strict tuple: (sem_cid, role)
            strict_set = frozenset(
                (node_to_body[b], role_of.get(b, "unknown")) for b in body
            )
            # Semantic-only: just sem_cid
            sem_set = frozenset(node_to_body[b] for b in body)
            # Role-pattern: counts of each role
            role_counts = Counter(role_of.get(b, "unknown") for b in body)
            rolepat = tuple(sorted(role_counts.items()))

            sigs_strict[strict_set] += 1
            sigs_semantic[sem_set] += 1
            sigs_rolepat[rolepat] += 1
            ri_pair = (node_to_risk[r_n], node_to_interv[i_n])
            sig_strict_RI_pairs[strict_set].add(ri_pair)
            sig_semantic_RI_pairs[sem_set].add(ri_pair)
            sig_rolepat_RI_pairs[rolepat].add(ri_pair)

    print(
        f"  paths total={n_total:,}  strict={n_strict:,} ({n_strict / max(n_total, 1) * 100:.2f}%)"
    )
    print(f"  unique strict-tuple frozensets: {len(sigs_strict):,}")
    print(f"  unique semantic-only frozensets: {len(sigs_semantic):,}")
    print(f"  unique role-pattern signatures : {len(sigs_rolepat):,}")

    # ─── Hamming families at each level ───────────────────────────────────────
    summary_rows = []
    detail_rows = []

    for level_name, sig_counter, ri_map, sig_to_str_fn in [
        ("strict_tuple", sigs_strict, sig_strict_RI_pairs, sig_to_str_strict),
        ("semantic_only", sigs_semantic, sig_semantic_RI_pairs, sig_to_str_semantic),
        ("role_pattern", sigs_rolepat, sig_rolepat_RI_pairs, role_pattern_to_str),
    ]:
        print(f"\n{'-' * 70}")
        print(f"LEVEL: {level_name}")
        print(f"{'-' * 70}")
        sig_list = list(sig_counter.keys())
        n = len(sig_list)
        if n == 0:
            print("  no frozensets — skipping")
            continue
        n_paths_per_sig = [sig_counter[s] for s in sig_list]
        print(f"  unique signatures: {n}")
        print(f"  total paths: {sum(n_paths_per_sig):,}")
        print(
            f"  multi-path signatures (n>=2): {sum(1 for c in n_paths_per_sig if c >= 2)}"
        )
        # Jaccard distance distribution (only for set-valued levels)
        if level_name != "role_pattern":
            jd = jaccard_distance_matrix(sig_list)
            jd_flat = jd[np.triu_indices(n, k=1)]
            if len(jd_flat) > 0:
                print(
                    f"  Jaccard distance — mean={jd_flat.mean():.4f} "
                    f"median={np.median(jd_flat):.4f}"
                )
        # Hamming families
        comps = hamming_families(sig_list, max_dist=args.max_hamming)
        n_multi = sum(1 for c in comps if len(c) > 1)
        print(f"  Hamming families at d<={args.max_hamming}: {len(comps):,}")
        print(f"  multi-frozenset families: {n_multi}")
        # Top 10 families
        print("  Top 10 families by total #paths:")
        for fam_idx, comp in enumerate(comps[:10]):
            tot_paths = sum(n_paths_per_sig[i] for i in comp)
            tot_RI = len(set().union(*[ri_map[sig_list[i]] for i in comp]))
            best_i = max(comp, key=lambda i: n_paths_per_sig[i])
            print(
                f"    fam {fam_idx:>3}  #frozensets={len(comp):>3}  "
                f"#paths={tot_paths:>4}  #R-I={tot_RI:>4}  "
                f"core={sig_to_str_fn(sig_list[best_i])[:55]}"
            )
            detail_rows.append(
                {
                    "cutoff": cutoff,
                    "level": level_name,
                    "family_id": fam_idx,
                    "n_frozensets": len(comp),
                    "n_paths_total": tot_paths,
                    "n_RI_pairs_total": tot_RI,
                    "core_signature": sig_to_str_fn(sig_list[best_i]),
                }
            )
        # Summary row
        summary_rows.append(
            {
                "cutoff": cutoff,
                "level": level_name,
                "n_strict_paths": n_strict,
                "n_unique_signatures": n,
                "n_multi_path_signatures": sum(1 for c in n_paths_per_sig if c >= 2),
                "n_hamming_families_d2": len(comps),
                "n_multi_frozenset_families": n_multi,
                "jaccard_mean": float(jd_flat.mean())
                if level_name != "role_pattern" and len(jd_flat) > 0
                else None,
                "jaccard_median": float(np.median(jd_flat))
                if level_name != "role_pattern" and len(jd_flat) > 0
                else None,
            }
        )

    # Save outputs
    out_dir = STEP4 / "step4_cluster_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"global_cutoff_summary_{suffix}.csv"
    summary_df.to_csv(summary_csv, index=False)
    detail_df = pd.DataFrame(detail_rows)
    detail_csv = out_dir / f"global_cutoff_top_families_{suffix}.csv"
    detail_df.to_csv(detail_csv, index=False)
    print(f"\nWrote: {summary_csv.name}")
    print(f"Wrote: {detail_csv.name}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
