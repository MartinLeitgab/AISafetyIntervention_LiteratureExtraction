"""phase2_routing_to_xlsx.py — Build xlsx snapshot of routing-pipeline state.

Reads:
  phase2_routing_active_catalog.json    — current HC + MC + axes
  phase2_routing_assignments.jsonl      — per-path (HC, MC, axes, confidence)

Writes:
  phase2_routing_combined.xlsx          — default OR override via --out

Sheets:
  summary           - catalog counts + axis distributions + source counts
  harm_classes      - long format, one row per (HC, member path)
  mechanism_classes - long format, one row per (MC, member path)
  axes              - axis definitions + value frequency
  paths             - one row per path with all dims (HC, MC, 6 axes, conf)

Class B. Run standalone or import build_routing_xlsx().
"""

from __future__ import annotations
import json
import pickle
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
ACTIVE_CATALOG_FP = STEP1 / "phase2_routing_active_catalog.json"
ASSIGNMENTS_FP = STEP1 / "phase2_routing_assignments.jsonl"
DEFAULT_OUT = STEP1 / "phase2_routing_combined.xlsx"


def load_paths_indexed():
    paths = {}
    fp = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
    with open(fp, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                d = json.loads(line)
                d["path_id"] = f"path_{i:05d}_dedup"
                paths[d["path_id"]] = d
    return paths


SUBTYPE_SHORT = {
    "problem_analysis": "pa",
    "theoretical_insight": "ti",
    "design_rationale": "dr",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
}


def render_traversal(p, node_attrs, max_len=120):
    if not p:
        return "(path not found)"
    lines = []
    nodes = p.get("path", [])
    cats = p.get("categories", [])
    for nid, cat in zip(nodes, cats):
        attrs = node_attrs.get(int(nid)) or node_attrs.get(nid) or {}
        name = (attrs.get("name") or "?")[:max_len]
        if cat == "risk":
            label = "risk"
        elif cat == "intervention":
            label = "interv"
        else:
            sub = attrs.get("subtype") or attrs.get("concept_category") or "body"
            label = f"body[{SUBTYPE_SHORT.get(sub, sub[:2])}]"
        lines.append(f"{label}|{name}")
    return " >> ".join(lines)


def build_routing_xlsx(out_path=None, paths_idx=None, node_attrs=None):
    out_path = Path(out_path) if out_path else DEFAULT_OUT
    if not ACTIVE_CATALOG_FP.exists():
        raise FileNotFoundError(
            f"\nERROR: {ACTIVE_CATALOG_FP} missing. "
            f"Run phase2_step5_opus_routing.py first to initialize the catalog.\n"
        )
    catalog = json.loads(ACTIVE_CATALOG_FP.read_text(encoding="utf-8"))
    print(
        f"catalog: {len(catalog['harm_classes'])} HC + "
        f"{len(catalog['mechanism_classes'])} MC + "
        f"{len(catalog['axes'])} axes"
    )

    assignments = []
    if ASSIGNMENTS_FP.exists():
        for line in ASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                assignments.append(json.loads(line))
    print(f"assignments: {len(assignments)} rows")

    if paths_idx is None:
        paths_idx = load_paths_indexed()
        print(f"  loaded {len(paths_idx)} deduped paths")
    if node_attrs is None:
        with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
            node_attrs = pickle.load(f)
        print(f"  loaded {len(node_attrs)} nodes")

    # Latest assignment per path_id wins
    latest = {a["path_id"]: a for a in assignments if a.get("path_id")}

    # Build groupings
    hc_members = defaultdict(list)
    mc_members = defaultdict(list)
    axis_value_counts = defaultdict(Counter)
    source_counts = Counter()
    path_rows = []

    for pid, r in latest.items():
        source_counts[r.get("source", "?")] += 1
        hc = r.get("harm_class_id")
        mc = r.get("mechanism_class_id")
        if hc:
            hc_members[hc].append(pid)
        if mc:
            mc_members[mc].append(pid)
        axes = r.get("axes", {}) or {}
        for axn, axv in axes.items():
            axis_value_counts[axn][axv] += 1
        p = paths_idx.get(pid)
        traversal = render_traversal(p, node_attrs)
        row = {
            "path_id": pid,
            "harm_class": hc or "",
            "mechanism_class": mc or "",
            "confidence": r.get("confidence") or "",
            "fit_note": r.get("fit_note") or "",
            "source": r.get("source", ""),
            "traversal": traversal,
        }
        for axn, axv in axes.items():
            row[f"axis_{axn}"] = axv
        path_rows.append(row)

    # HC long-format sheet
    hc_rows = []
    for h in catalog["harm_classes"]:
        gid = h["class_id"]
        flag = " [CAP-GAP]" if h.get("is_capability_gap") else ""
        pids = hc_members.get(gid, [])
        if not pids:
            hc_rows.append(
                {
                    "class_id": gid,
                    "class_name": h["class_name"] + flag,
                    "description": h.get("class_description", ""),
                    "n_paths": 0,
                    "path_id": "",
                    "traversal": "",
                }
            )
        else:
            for pid in pids:
                p = paths_idx.get(pid)
                hc_rows.append(
                    {
                        "class_id": gid,
                        "class_name": h["class_name"] + flag,
                        "description": h.get("class_description", ""),
                        "n_paths": len(pids),
                        "path_id": pid,
                        "traversal": render_traversal(p, node_attrs),
                    }
                )

    # MC long-format sheet
    mc_rows = []
    for m in catalog["mechanism_classes"]:
        gid = m["class_id"]
        pids = mc_members.get(gid, [])
        if not pids:
            mc_rows.append(
                {
                    "class_id": gid,
                    "class_name": m["class_name"],
                    "description": m.get("class_description", ""),
                    "n_paths": 0,
                    "path_id": "",
                    "traversal": "",
                }
            )
        else:
            for pid in pids:
                p = paths_idx.get(pid)
                mc_rows.append(
                    {
                        "class_id": gid,
                        "class_name": m["class_name"],
                        "description": m.get("class_description", ""),
                        "n_paths": len(pids),
                        "path_id": pid,
                        "traversal": render_traversal(p, node_attrs),
                    }
                )

    # Axes sheet
    ax_rows = []
    for ax in catalog["axes"]:
        axn = ax["axis_name"]
        for v in ax.get("values", []):
            ax_rows.append(
                {
                    "axis_name": axn,
                    "axis_kind": ax["axis_kind"],
                    "value": v,
                    "n_assigned": axis_value_counts[axn].get(v, 0),
                }
            )
        # OTHER:* free-text emergent values
        for v, cnt in axis_value_counts[axn].items():
            if v not in ax.get("values", []):
                ax_rows.append(
                    {
                        "axis_name": axn,
                        "axis_kind": ax["axis_kind"],
                        "value": v + " (emergent)",
                        "n_assigned": cnt,
                    }
                )

    # Summary
    n_total_routed = len(latest)
    df_summary = pd.DataFrame(
        [
            {"item": "harm_classes total", "value": len(catalog["harm_classes"])},
            {
                "item": "harm_classes with >=1 member",
                "value": sum(1 for v in hc_members.values() if v),
            },
            {
                "item": "mechanism_classes total",
                "value": len(catalog["mechanism_classes"]),
            },
            {
                "item": "mechanism_classes with >=1 member",
                "value": sum(1 for v in mc_members.values() if v),
            },
            {"item": "axes total", "value": len(catalog["axes"])},
            {"item": "paths assigned", "value": n_total_routed},
            {"item": "deduped corpus", "value": len(paths_idx)},
            {
                "item": "coverage %",
                "value": round(100 * n_total_routed / len(paths_idx), 1)
                if paths_idx
                else 0,
            },
        ]
        + [
            {"item": f"source:{k}", "value": v}
            for k, v in sorted(source_counts.items())
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {out_path.name} ...")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_summary.to_excel(xw, sheet_name="summary", index=False)
        pd.DataFrame(hc_rows).to_excel(xw, sheet_name="harm_classes", index=False)
        pd.DataFrame(mc_rows).to_excel(xw, sheet_name="mechanism_classes", index=False)
        pd.DataFrame(ax_rows).to_excel(xw, sheet_name="axes", index=False)
        pd.DataFrame(path_rows).to_excel(xw, sheet_name="paths", index=False)
    print(f"wrote {out_path}")
    print(f"  summary: {len(df_summary)} rows")
    print(
        f"  harm_classes: {len(hc_rows)} rows; mechanism_classes: {len(mc_rows)}; "
        f"axes: {len(ax_rows)}; paths: {len(path_rows)}"
    )
    return {
        "out_path": str(out_path),
        "n_harm_classes": len(catalog["harm_classes"]),
        "n_mech_classes": len(catalog["mechanism_classes"]),
        "n_paths": n_total_routed,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    build_routing_xlsx(out_path=args.out)
