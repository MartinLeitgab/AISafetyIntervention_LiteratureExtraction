"""build_pilot_v2_xlsx.py — Build xlsx audit view of pilot v2 catalog + assignments.

Class B (no LLM). Reads:
  phase2_pilot_v2_100paths_discovery.json (canonical v2 output)
  phase2_pilot_v2_100paths_assignments.jsonl (flattened)

Writes:
  phase2_pilot_v2_audit.xlsx
"""

from __future__ import annotations
import json
import sys
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M
from phase2_routing_to_xlsx import render_traversal, load_paths_indexed

PILOT_V2_FP = M.STEP1 / "phase2_pilot_v2_100paths_discovery.json"
OUT_FP = M.STEP1 / "phase2_pilot_v2_audit.xlsx"


def main():
    d = json.loads(PILOT_V2_FP.read_text(encoding="utf-8"))
    raw = d["raw_output"]
    paths_idx = load_paths_indexed()
    with open(M.STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)

    # Build assignment lookup
    asg_by_pid = {a["path_id"]: a for a in raw["assignments"]}

    # HC sheet
    hc_rows = []
    for h in raw["harm_classes"]:
        gid = h["class_id"]
        flag = " [CAP-GAP]" if h.get("is_capability_gap") else ""
        members = [
            pid for pid, a in asg_by_pid.items() if a.get("harm_class_id") == gid
        ]
        if not members:
            hc_rows.append(
                {
                    "class_id": gid,
                    "class_name": h["class_name"] + flag,
                    "description": h.get("class_description", ""),
                    "expected_n": h.get("expected_n_paths"),
                    "actual_n": 0,
                    "path_id": "",
                    "fit_score": "",
                    "traversal": "",
                }
            )
        else:
            for pid in members:
                p = paths_idx.get(pid)
                a = asg_by_pid[pid]
                hc_rows.append(
                    {
                        "class_id": gid,
                        "class_name": h["class_name"] + flag,
                        "description": h.get("class_description", ""),
                        "expected_n": h.get("expected_n_paths"),
                        "actual_n": len(members),
                        "path_id": pid,
                        "fit_score": a.get("fit_score"),
                        "confidence": a.get("confidence"),
                        "fit_note": a.get("fit_note", ""),
                        "traversal": render_traversal(p, na),
                    }
                )

    # MC sheet
    mc_rows = []
    for m in raw["mechanism_classes"]:
        gid = m["class_id"]
        members = [
            pid for pid, a in asg_by_pid.items() if a.get("mechanism_class_id") == gid
        ]
        if not members:
            mc_rows.append(
                {
                    "class_id": gid,
                    "class_name": m["class_name"],
                    "description": m.get("class_description", ""),
                    "expected_n": m.get("expected_n_paths"),
                    "actual_n": 0,
                    "path_id": "",
                    "traversal": "",
                }
            )
        else:
            for pid in members:
                p = paths_idx.get(pid)
                a = asg_by_pid[pid]
                mc_rows.append(
                    {
                        "class_id": gid,
                        "class_name": m["class_name"],
                        "description": m.get("class_description", ""),
                        "expected_n": m.get("expected_n_paths"),
                        "actual_n": len(members),
                        "path_id": pid,
                        "fit_score": a.get("fit_score"),
                        "confidence": a.get("confidence"),
                        "fit_note": a.get("fit_note", ""),
                        "traversal": render_traversal(p, na),
                    }
                )

    # Unassigned sheet (separately)
    unassigned_rows = []
    for pid, a in asg_by_pid.items():
        hc = a.get("harm_class_id")
        mc = a.get("mechanism_class_id")
        if isinstance(hc, dict) and hc.get("unassigned"):
            p = paths_idx.get(pid)
            unassigned_rows.append(
                {
                    "path_id": pid,
                    "side": "harm_class",
                    "reason": hc.get("reason", ""),
                    "fit_score": a.get("fit_score"),
                    "fit_note": a.get("fit_note", ""),
                    "traversal": render_traversal(p, na),
                }
            )
        if isinstance(mc, dict) and mc.get("unassigned"):
            p = paths_idx.get(pid)
            unassigned_rows.append(
                {
                    "path_id": pid,
                    "side": "mechanism_class",
                    "reason": mc.get("reason", ""),
                    "fit_score": a.get("fit_score"),
                    "fit_note": a.get("fit_note", ""),
                    "traversal": render_traversal(p, na),
                }
            )

    # Axes sheet
    ax_value_counts = defaultdict(Counter)
    for a in raw["assignments"]:
        for axn, axv in (a.get("axis_values") or {}).items():
            ax_value_counts[axn][axv] += 1
    ax_rows = []
    for ax in raw["axes"]:
        axn = ax["axis_name"]
        for v in ax.get("values", []):
            ax_rows.append(
                {
                    "axis_name": axn,
                    "axis_kind": ax["axis_kind"],
                    "value": v,
                    "n_assigned": ax_value_counts[axn].get(v, 0),
                }
            )
        for v, cnt in ax_value_counts[axn].items():
            if v not in ax.get("values", []):
                ax_rows.append(
                    {
                        "axis_name": axn,
                        "axis_kind": ax["axis_kind"],
                        "value": v + " (emergent)",
                        "n_assigned": cnt,
                    }
                )

    # Paths sheet (one row per path)
    path_rows = []
    for pid, a in asg_by_pid.items():
        hc = a.get("harm_class_id")
        mc = a.get("mechanism_class_id")
        hc_id = hc if not isinstance(hc, dict) else "UNASSIGNED"
        mc_id = mc if not isinstance(mc, dict) else "UNASSIGNED"
        row = {
            "path_id": pid,
            "harm_class": hc_id,
            "mechanism_class": mc_id,
            "confidence": a.get("confidence"),
            "fit_score": a.get("fit_score"),
            "fit_note": a.get("fit_note", ""),
            "harm_target_evidence": a.get("harm_target_evidence", ""),
            "traversal": render_traversal(paths_idx.get(pid), na),
        }
        for axn, axv in (a.get("axis_values") or {}).items():
            row[f"axis_{axn}"] = axv
        path_rows.append(row)

    # Summary
    summary = pd.DataFrame(
        [
            {"item": "n_harm_classes", "value": len(raw["harm_classes"])},
            {"item": "n_mech_classes", "value": len(raw["mechanism_classes"])},
            {"item": "n_axes", "value": len(raw["axes"])},
            {"item": "n_paths_routed", "value": len(raw["assignments"])},
            {"item": "n_unassigned_rows", "value": len(unassigned_rows)},
            {
                "item": "architecture_critique",
                "value": raw.get("architecture_critique", "")[:300] + "...",
            },
        ]
    )

    print(f"writing {OUT_FP.name} ...", flush=True)
    with pd.ExcelWriter(OUT_FP, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        pd.DataFrame(hc_rows).to_excel(xw, sheet_name="harm_classes", index=False)
        pd.DataFrame(mc_rows).to_excel(xw, sheet_name="mechanism_classes", index=False)
        pd.DataFrame(unassigned_rows).to_excel(xw, sheet_name="unassigned", index=False)
        pd.DataFrame(ax_rows).to_excel(xw, sheet_name="axes", index=False)
        pd.DataFrame(path_rows).to_excel(xw, sheet_name="paths", index=False)
    print(f"wrote {OUT_FP}", flush=True)
    print(
        f"  HC rows: {len(hc_rows)}, MC rows: {len(mc_rows)}, "
        f"unassigned rows: {len(unassigned_rows)}, paths: {len(path_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
