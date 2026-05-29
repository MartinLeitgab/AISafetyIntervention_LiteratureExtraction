"""build_seed_only_xlsx.py — render the original 200-path Opus seed-only state
as an xlsx for paper-reproducibility audit.

Reads `phase2_doublet_seed_catalog.json` DIRECTLY (not via the remap-applying
merged jsonl), so group_ids reflect the ORIGINAL Opus assignment — not any
post-REVIEW renames/merges/splits. The 34 RG + 119 MG catalog + 200 assignments
are written to phase2_doublet_review_seed_only_orig.xlsx alongside the other
review xlsx variants.

This is the workshop-reviewer artifact for "what Opus did on the seed sample
before any Sonnet routing or any Opus REVIEW." Class B, no LLM calls.
"""

import json
import pickle
import pandas as pd
from collections import defaultdict
from phase2_step4_phase2_doublet_to_xlsx import (
    STEP1,
    build_edge_lookup,
    load_paths_indexed,
    render_traversal,
)

OUT_FP = STEP1 / "phase2_doublet_review_seed_only_orig.xlsx"
SEED_FP = STEP1 / "phase2_doublet_seed_catalog.json"


def main():
    seed = json.loads(SEED_FP.read_text(encoding="utf-8"))
    print(
        f"seed catalog: {len(seed['risk_groups'])} RG + "
        f"{len(seed['mechanism_groups'])} MG + {len(seed['assignments'])} assignments"
    )
    paths_idx = load_paths_indexed()
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    with open(STEP1 / "graph_edge_data.pkl", "rb") as f:
        edge_data = pickle.load(f)
    edge_lookup = build_edge_lookup(edge_data)

    rg_members = defaultdict(list)
    mg_members = defaultdict(list)
    path_rows = []
    for a in seed["assignments"]:
        pid = a["path_id"]
        rg = a["risk_group_id"]
        mg = a["mechanism_group_id"]
        if rg:
            rg_members[rg].append(pid)
        if mg:
            mg_members[mg].append(pid)
        p = paths_idx.get(pid)
        traversal = (
            render_traversal(p, node_attrs, edge_lookup) if p else "(path not found)"
        )
        path_rows.append(
            {
                "path_id": pid,
                "risk_group_orig": rg or "",
                "mechanism_group_orig": mg or "",
                "source": "seed (orig Opus)",
                "traversal": traversal,
                "notes": "",
            }
        )

    def long_rows(groups, members):
        rows = []
        for g in groups:
            gid = g["group_id"]
            pids = members.get(gid, [])
            if not pids:
                rows.append(
                    {
                        "group_id": gid,
                        "group_name": g["group_name"],
                        "description": g["group_description"],
                        "n_paths": 0,
                        "path_id": "",
                        "path_traversal": "",
                    }
                )
            else:
                for pid in pids:
                    p = paths_idx.get(pid)
                    traversal = (
                        render_traversal(p, node_attrs, edge_lookup)
                        if p
                        else "(path not found)"
                    )
                    rows.append(
                        {
                            "group_id": gid,
                            "group_name": g["group_name"],
                            "description": g["group_description"],
                            "n_paths": len(pids),
                            "path_id": pid,
                            "path_traversal": traversal,
                        }
                    )
        return rows

    df_rg = pd.DataFrame(long_rows(seed["risk_groups"], rg_members))
    df_mg = pd.DataFrame(long_rows(seed["mechanism_groups"], mg_members))
    df_paths = pd.DataFrame(path_rows)
    df_summary = pd.DataFrame(
        [
            {"item": "risk_groups (orig Opus seed)", "n": len(seed["risk_groups"])},
            {
                "item": "mechanism_groups (orig Opus seed)",
                "n": len(seed["mechanism_groups"]),
            },
            {"item": "assigned paths", "n": len(seed["assignments"])},
            {
                "item": "RG with >=1 member",
                "n": sum(1 for v in rg_members.values() if v),
            },
            {
                "item": "MG with >=1 member",
                "n": sum(1 for v in mg_members.values() if v),
            },
            {
                "item": "snapshot epoch",
                "n": "2026-05-15 Opus seed-gen (pre-Sonnet, pre-REVIEW)",
            },
        ]
    )

    print(f"writing {OUT_FP.name} ...")
    with pd.ExcelWriter(OUT_FP, engine="openpyxl") as xw:
        df_summary.to_excel(xw, sheet_name="summary", index=False)
        df_rg.to_excel(xw, sheet_name="risk_groups", index=False)
        df_mg.to_excel(xw, sheet_name="mechanism_groups", index=False)
        df_paths.to_excel(xw, sheet_name="paths", index=False)
    print(f"wrote {OUT_FP}")
    print(f"  summary: {len(df_summary)} rows")
    print(f"  risk_groups: {len(df_rg)} rows")
    print(f"  mechanism_groups: {len(df_mg)} rows")
    print(f"  paths: {len(df_paths)} rows")


if __name__ == "__main__":
    main()
