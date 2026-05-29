"""
phase2_step4_phase2_review_combine.py — Combine Pass-A review (200 sample nodes) with
smoke batch decisions into per-subtype review files showing (group_name -> member nodes).

Inputs (per NR subtype st in {pa, ti, dr, im, va, interv}):
  - phase2_full_vpn_seed_per_subtype_nr/seed_<st>.json           (Pass A: catalog)
  - phase2_full_vpn_review_nr/review_<st>.json                   (Pass-A review: 200 sample -> groups)
  - phase2_full_vpn_batches/nr/batch_smoke_<st>.json             (Smoke: 80 random -> groups)

Output (per subtype):
  - phase2_full_vpn_review_combined_nr/combined_<st>.json

No LLM calls. Pure JSON merge. Fast.
"""

import json
import re
import sys
from pathlib import Path

# Strip "G123. " prefix the LLM sometimes prepends to group names (artifact of catalog display)
G_PREFIX_RE = re.compile(r"^G\d+\.\s+")


def normalize_group_name(name: str) -> str:
    if not name:
        return name
    return G_PREFIX_RE.sub("", name.strip())


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"

SUBTYPE_SHORT = {
    "risk": "risk",
    "problem_analysis": "pa",
    "theoretical_insight": "ti",
    "design_rationale": "dr",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
    "intervention": "interv",
}
SHORT_TO_LONG = {v: k for k, v in SUBTYPE_SHORT.items()}

POOL_SUBTYPES = {
    "nr": [
        "problem_analysis",
        "theoretical_insight",
        "design_rationale",
        "implementation_mechanism",
        "validation_evidence",
        "intervention",
    ],
    "risk": ["risk"],
}


def combine_subtype(pool: str, st_long: str, st_short: str, node_attrs: dict = None):
    """Combine seed + review + smoke for one subtype. Returns combined dict or None."""
    seed_file = (
        STEP1 / f"phase2_full_vpn_seed_per_subtype_{pool}" / f"seed_{st_short}.json"
    )
    review_file = STEP1 / f"phase2_full_vpn_review_{pool}" / f"review_{st_short}.json"
    smoke_file = STEP1 / f"phase2_full_vpn_batches/{pool}/batch_smoke_{st_short}.json"

    if not seed_file.exists():
        print(f"  [{st_short}] SKIP — seed file missing: {seed_file.name}")
        return None

    seed = json.loads(seed_file.read_text(encoding="utf-8"))
    review = (
        json.loads(review_file.read_text(encoding="utf-8"))
        if review_file.exists()
        else None
    )
    smoke = (
        json.loads(smoke_file.read_text(encoding="utf-8"))
        if smoke_file.exists()
        else None
    )

    # Build (group_name -> {description, members}) from seed catalog
    groups = {}
    for g in seed.get("groups", []):
        groups[g["name"]] = {
            "description": g.get("description", ""),
            "members_from_seed_review": [],  # 200-node Pass-A review
            "members_from_smoke": [],  # 80-node smoke batch
        }

    # Add smoke "new" proposed groups (not in seed catalog)
    if smoke:
        for d in smoke.get("decisions", []):
            if d.get("decision") == "new" and d.get("group_name"):
                gname_norm = normalize_group_name(d["group_name"])
                if gname_norm and gname_norm not in groups:
                    groups[gname_norm] = {
                        "description": d.get("group_description", "")
                        or "(no description — newly proposed in smoke)",
                        "members_from_seed_review": [],
                        "members_from_smoke": [],
                    }

    # Populate Pass-A review members (200 nodes)
    if review:
        for entry in review.get("review", []):
            gname = normalize_group_name(entry.get("group_name", ""))
            if gname in groups:
                for m in entry.get("members", []):
                    groups[gname]["members_from_seed_review"].append(
                        {
                            "node_id": m.get("node_id"),
                            "name": m.get("name", ""),
                        }
                    )

    # Populate smoke members (80 nodes)
    no_fit_smoke = []
    if smoke:
        for d in smoke.get("decisions", []):
            dec = d.get("decision")
            gname = normalize_group_name(d.get("group_name", ""))
            nid = d.get("node_id")
            if dec in ("seed", "new") and gname in groups:
                node_name = ""
                if node_attrs is not None and nid is not None:
                    a = node_attrs.get(int(nid)) or node_attrs.get(nid) or {}
                    node_name = (a.get("name") or "").strip()
                # Fall back to name stored in decision (if main script was updated)
                if not node_name:
                    node_name = d.get("name", "")
                groups[gname]["members_from_smoke"].append(
                    {
                        "node_id": nid,
                        "name": node_name,
                        "decision_type": dec,
                        "confidence": d.get("confidence"),
                    }
                )
            elif dec == "residual":
                node_name = ""
                if node_attrs is not None and nid is not None:
                    a = node_attrs.get(int(nid)) or node_attrs.get(nid) or {}
                    node_name = (a.get("name") or "").strip()
                no_fit_smoke.append({"node_id": nid, "name": node_name})

    # Build no-fit list from seed-review
    no_fit_seed_review = []
    if review:
        for m in review.get("no_fit_nodes", []):
            no_fit_seed_review.append(m)

    # Compute summary
    n_groups = len(groups)
    n_groups_with_members = sum(
        1
        for g in groups.values()
        if g["members_from_seed_review"] or g["members_from_smoke"]
    )
    n_seed_review_total = sum(
        len(g["members_from_seed_review"]) for g in groups.values()
    )
    n_smoke_total = sum(len(g["members_from_smoke"]) for g in groups.values())

    # Dedup by node_id across sources (smoke RNG overlaps with Pass-A review RNG, so
    # the same node can be assigned twice — once in each pass — to the same group).
    # When a node appears in both sources, merge to a single entry with source = "both".
    dedup_groups = {}
    for gname, g in groups.items():
        by_id = {}
        for m in g["members_from_seed_review"]:
            nid = m.get("node_id")
            by_id[nid] = {
                "node_id": nid,
                "name": m.get("name", ""),
                "sources": ["pass_a_review"],
            }
        for m in g["members_from_smoke"]:
            nid = m.get("node_id")
            if nid in by_id:
                by_id[nid]["sources"].append("smoke")
                by_id[nid]["smoke_decision"] = m.get("decision_type")
                by_id[nid]["smoke_confidence"] = m.get("confidence")
                # Use name from smoke if review didn't have one
                if not by_id[nid]["name"]:
                    by_id[nid]["name"] = m.get("name", "")
            else:
                by_id[nid] = {
                    "node_id": nid,
                    "name": m.get("name", ""),
                    "sources": ["smoke"],
                    "smoke_decision": m.get("decision_type"),
                    "smoke_confidence": m.get("confidence"),
                }
        dedup_groups[gname] = {
            "description": g["description"],
            "unique_members": list(by_id.values()),
            "n_only_review": sum(
                1 for v in by_id.values() if v["sources"] == ["pass_a_review"]
            ),
            "n_only_smoke": sum(1 for v in by_id.values() if v["sources"] == ["smoke"]),
            "n_both": sum(
                1
                for v in by_id.values()
                if set(v["sources"]) == {"pass_a_review", "smoke"}
            ),
        }
    # Sort groups by distinct member count (desc)
    sorted_groups = sorted(
        dedup_groups.items(),
        key=lambda kv: -len(kv[1]["unique_members"]),
    )

    combined = {
        "pool": pool,
        "subtype": st_long,
        "n_groups_total": n_groups,
        "n_groups_with_members": n_groups_with_members,
        "n_groups_empty": n_groups - n_groups_with_members,
        "n_members_from_pass_a_review": n_seed_review_total,
        "n_members_from_smoke_batch": n_smoke_total,
        "n_no_fit_pass_a_review": len(no_fit_seed_review),
        "n_residual_smoke": len(no_fit_smoke),
        "groups": [
            {
                "group_name": name,
                "description": g["description"],
                "n_distinct_members": len(g["unique_members"]),
                "n_only_review": g["n_only_review"],
                "n_only_smoke": g["n_only_smoke"],
                "n_both_sources_same_decision": g["n_both"],
                "members": g["unique_members"],
            }
            for name, g in sorted_groups
        ],
        "no_fit_pass_a_review_nodes": no_fit_seed_review,
        "residual_smoke_nodes": no_fit_smoke,
    }

    return combined


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["nr", "risk", "both"], default="both")
    args = ap.parse_args()
    pools = ["nr", "risk"] if args.pool == "both" else [args.pool]

    # Load node_attrs once (shared across pools)
    print("loading graph_node_attributes.pkl ...")
    import pickle
    import time

    t0 = time.time()
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  loaded {len(node_attrs)} nodes in {time.time() - t0:.1f}s")

    for pool in pools:
        out_dir = STEP1 / f"phase2_full_vpn_review_combined_{pool}"
        out_dir.mkdir(parents=True, exist_ok=True)
        subtypes = POOL_SUBTYPES[pool]
        print(
            f"\n{'=' * 60}\nCombining Pass-A review + smoke for {pool} pool, {len(subtypes)} subtypes\n{'=' * 60}"
        )

        summary_table = []
        for st in subtypes:
            st_short = SUBTYPE_SHORT[st]
            print(f"\n--- {st} ({st_short}) ---")
            combined = combine_subtype(pool, st, st_short, node_attrs=node_attrs)
            if combined is None:
                continue
            out_file = out_dir / f"combined_{st_short}.json"
            out_file.write_text(json.dumps(combined, indent=2), encoding="utf-8")
            print(
                f"  groups (total / with members / empty):  "
                f"{combined['n_groups_total']} / "
                f"{combined['n_groups_with_members']} / "
                f"{combined['n_groups_empty']}"
            )
            print(
                f"  members (pass-A review / smoke):  "
                f"{combined['n_members_from_pass_a_review']} / "
                f"{combined['n_members_from_smoke_batch']}"
            )
            print(
                f"  residual (pass-A review no_fit / smoke residual):  "
                f"{combined['n_no_fit_pass_a_review']} / "
                f"{combined['n_residual_smoke']}"
            )
            print(f"  wrote {out_file}")
            print("  top-5 groups by distinct member count:")
            for g in combined["groups"][:5]:
                print(f"    [{g['n_distinct_members']:3d}] {g['group_name']}")
            summary_table.append(
                {
                    "subtype": st,
                    "n_groups": combined["n_groups_total"],
                    "n_with_members": combined["n_groups_with_members"],
                    "n_review_assigned": combined["n_members_from_pass_a_review"],
                    "n_smoke_assigned": combined["n_members_from_smoke_batch"],
                    "n_review_no_fit": combined["n_no_fit_pass_a_review"],
                    "n_smoke_residual": combined["n_residual_smoke"],
                }
            )

        summary_path = out_dir / "_summary.json"
        summary_path.write_text(json.dumps(summary_table, indent=2), encoding="utf-8")
        print(f"\nsummary: {summary_path}")
        print(f"DONE pool={pool}. Files in: {out_dir}")


if __name__ == "__main__":
    main()
