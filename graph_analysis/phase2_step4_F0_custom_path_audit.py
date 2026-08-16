"""
phase2_step4_F0_custom_path_audit.py [rev8]

Audits paths_custom_*.jsonl for the custom-mode invariants and reports
retention vs the original unconstrained paths.

Invariants checked:
  1. Every path starts with a risk node (cat_path[0] == "risk").
  2. Every path ends with an intervention node (cat_path[-1] == "intervention").
  3. No risk nodes appear in the middle (cat_path[1:-1] has zero "risk").
  4. No intervention nodes appear in the middle (cat_path[1:-1] has zero
     "intervention").
  5. Every middle node has a body subtype (problem_analysis,
     theoretical_insight, design_rationale, implementation_mechanism,
     validation_evidence).

Retention metrics (per threshold):
  - n_paths_unconstrained vs n_paths_custom (count + percentage)
  - distinct (risk_cluster, interv_cluster) R->I pair coverage delta

Output:
  step4_finalanalysis/step4_connectivity/custom_mode_audit_report.csv
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
PATHS_DIR_OLD = ROOT / "phase1_rawpathsfiles"
PATHS_DIR_NEW = ROOT / "phase1_otherrawdata"  # where new run writes
OUT_DIR = ROOT / "phase2_results/step4_finalanalysis/step4_connectivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BODY_SUBTYPES = {
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
}
THRESHOLDS = [
    ("EDGE", "edge_only"),
    ("0.80", "sim0.8"),
    ("0.85", "sim0.85"),
    ("0.90", "sim0.9"),
    ("0.95", "sim0.95"),
]


def audit_path_file(path):
    """Return summary dict for a paths_*.jsonl file."""
    if not path.exists():
        return None
    n_total = 0
    n_violations = Counter()
    middle_cats = Counter()
    ri_pairs = set()
    with open(path) as f:
        for line in f:
            n_total += 1
            obj = json.loads(line)
            cats = obj.get("categories", [])
            nodes = obj.get("path", [])
            if not cats or len(cats) < 2:
                n_violations["too_short"] += 1
                continue
            # Invariant checks
            if cats[0] != "risk":
                n_violations["start_not_risk"] += 1
            if cats[-1] != "intervention":
                n_violations["end_not_intervention"] += 1
            for c in cats[1:-1]:
                middle_cats[c] += 1
                if c == "risk":
                    n_violations["risk_in_middle"] += 1
                elif c == "intervention":
                    n_violations["intervention_in_middle"] += 1
                elif c not in BODY_SUBTYPES:
                    n_violations["non_body_in_middle"] += 1
            # R->I pair (using node IDs as proxy; cluster lookup happens
            # downstream — this is enough to compare coverage)
            ri_pairs.add((nodes[0], nodes[-1]))
    return {
        "n_paths": n_total,
        "n_unique_ri_pairs": len(ri_pairs),
        "violations": dict(n_violations),
        "middle_cat_dist": dict(middle_cats),
    }


def main():
    rows = []
    print("=" * 80)
    print("CUSTOM-MODE PATH AUDIT — rev8")
    print("=" * 80)
    for thr_name, thr_label in THRESHOLDS:
        unc_path = PATHS_DIR_OLD / f"paths_unconstrained_{thr_label}.jsonl"
        # Custom output: try both possible locations
        cus_path = PATHS_DIR_NEW / f"paths_custom_{thr_label}.jsonl"
        if not cus_path.exists():
            cus_path_alt = PATHS_DIR_OLD / f"paths_custom_{thr_label}.jsonl"
            if cus_path_alt.exists():
                cus_path = cus_path_alt

        print(f"\n=== Threshold: {thr_name} ===")
        unc = audit_path_file(unc_path)
        cus = audit_path_file(cus_path)

        if unc is None:
            print(f"  unconstrained: file missing ({unc_path})")
            unc_n = 0
            unc_pairs = 0
        else:
            unc_n = unc["n_paths"]
            unc_pairs = unc["n_unique_ri_pairs"]
            print(f"  unconstrained: {unc_n:,} paths, {unc_pairs:,} unique R->I pairs")
            if unc["violations"]:
                print(f"    violations: {unc['violations']}")

        if cus is None:
            print(f"  custom: file missing ({cus_path})")
            cus_n = 0
            cus_pairs = 0
            cus_violations = {}
        else:
            cus_n = cus["n_paths"]
            cus_pairs = cus["n_unique_ri_pairs"]
            cus_violations = cus["violations"]
            print(f"  custom:        {cus_n:,} paths, {cus_pairs:,} unique R->I pairs")
            print(f"    violations: {cus_violations or '(none — clean)'}")
            print(f"    middle cat dist: {cus['middle_cat_dist']}")

        retention_paths = f"{100 * cus_n / unc_n:.1f}%" if unc_n else "n/a"
        retention_pairs = f"{100 * cus_pairs / unc_pairs:.1f}%" if unc_pairs else "n/a"
        clean = "yes" if cus is not None and not cus_violations else "no"
        print(f"  retention: paths={retention_paths}, R->I pairs={retention_pairs}")
        print(f"  invariants clean: {clean}")

        rows.append(
            {
                "threshold": thr_name,
                "label": thr_label,
                "unconstrained_paths": unc_n,
                "custom_paths": cus_n,
                "retention_paths": retention_paths,
                "unconstrained_ri_pairs": unc_pairs,
                "custom_ri_pairs": cus_pairs,
                "retention_ri_pairs": retention_pairs,
                "custom_invariants_clean": clean,
                "custom_violations": str(cus_violations),
            }
        )

    out_csv = OUT_DIR / "custom_mode_audit_report.csv"
    import csv

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
