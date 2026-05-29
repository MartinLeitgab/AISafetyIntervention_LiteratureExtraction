"""
phase2_step4_phase2_doublet_to_xlsx.py — Build the doublet review spreadsheet
for §19.13.

CANONICAL INPUTS (no fallbacks per user-level CLAUDE.md fail-fast rule):
  phase2_doublet_assignments.jsonl   — merged path -> (RG, MG) assignments
                                       (rebuilt after every Pass B batch + after seed/smoke)
  phase2_doublet_seed_catalog.json   — RG + MG catalog (definitions)
  phase2_doublet_active_catalog.json — latest catalog if Pass B has added groups (optional;
                                       only used WHEN PRESENT to reflect Pass B's expanded
                                       catalog; never falls back to seed if Pass B has run)
  paths_hopwise_v4_edge_only.jsonl   — path node sequences (extraction-LLM order)
  graph_node_attributes.pkl          — node names
  graph_edge_data.pkl                — edge subtypes / descriptions

The script CRASHES if the merged jsonl OR the seed catalog is missing. It does NOT
fall back to legacy review.json or to manual-baseline mode.

Output:
  phase2_doublet_review_combined_v2.xlsx

Sheets:
  summary            — group counts per pool + assignment source breakdown
  risk_groups        — one row per RG: group_id, group_name, description, n_paths, paths_text
  mechanism_groups   — one row per MG: same shape
  paths              — one row per assigned path: path_id, risk_group, mechanism_group,
                       source, traversal (full names + concept subtype per node), notes

Path traversal format (extraction-LLM order, full names, concept subtype per node):
  <node1 name> [risk] --[edge_subtype]--> <node2 name> [pa] --[edge_subtype]--> ...

CALLED FROM TWO PLACES:
  1. CLI: `python phase2_step4_phase2_doublet_to_xlsx.py` — standalone re-build.
  2. After each Pass B batch from run_full() in phase2_step4_phase2_doublet_llm_grouping.py,
     via import: `build_xlsx_from_disc(node_attrs=..., paths_idx=..., edge_lookup=...)`.
     Passing in already-loaded data avoids re-loading 200k nodes + 1.7M edges per batch.
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
PATHS_FILE = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"

CATALOG_FP = STEP1 / "phase2_doublet_seed_catalog.json"
ACTIVE_CATALOG_FP = (
    STEP1 / "phase2_doublet_active_catalog.json"
)  # written during Pass B
MERGED_JSONL_FP = STEP1 / "phase2_doublet_assignments.jsonl"
REVIEW_FP = STEP1 / "phase2_doublet_review.json"  # legacy
OUT_FP = STEP1 / "phase2_doublet_review_combined_v2.xlsx"

MAX_DESC_CHARS = 400


def truncate(s, n):
    s = (s or "").strip().replace("\n", " ").replace("\r", " ")
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "..."


def load_paths_indexed():
    """Returns {path_id -> {nodes: [...], categories: [...]}} keyed by path_00001, etc."""
    out = {}
    with open(PATHS_FILE, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = f"path_{lineno:05d}"
            out[pid] = {
                "nodes": d["path"],
                "categories": d["categories"],
                "edge_types": d.get("edge_types", []),
                "length": d.get("length"),
            }
    return out


def build_edge_lookup(edge_data):
    """Build (source, target) -> {subtype, description} for EDGE-typed edges only."""
    lookup = {}
    for e in edge_data:
        if str(e.get("type", "")).upper() != "EDGE":
            continue
        s, t = e.get("source"), e.get("target")
        if s is None or t is None:
            continue
        lookup[(int(s), int(t))] = {
            "subtype": e.get("subtype", ""),
            "description": e.get("description", ""),
        }
    return lookup


SUBTYPE_SHORT = {
    "risk": "risk",
    "problem_analysis": "pa",
    "theoretical_insight": "ti",
    "design_rationale": "dr",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
    "intervention": "interv",
}


def render_traversal(path_rec, node_attrs, edge_lookup):
    """Return a single-string traversal of one path:
       '<n1 name> [subtype] --[edge_subtype]--> <n2 name> [subtype] --[edge_subtype]--> ...'
    - Uses extraction-LLM order (path_rec['nodes'] is in path-traversal order).
    - Node NAMES are NOT truncated (full names).
    - Each node tagged with its concept subtype short code (risk|pa|ti|dr|im|va|interv).
    """
    nodes = path_rec["nodes"]
    cats = path_rec.get("categories", []) or []
    pieces = []
    for i, nid in enumerate(nodes):
        attrs = node_attrs.get(int(nid)) or node_attrs.get(nid) or {}
        name = (attrs.get("name", "?") or "").strip().replace("\n", " ")
        # Determine concept subtype: positional category disambiguates risk vs intervention vs body;
        # for body, fall back to node attrs subtype/concept_category.
        pos_cat = cats[i] if i < len(cats) else ""
        if pos_cat == "risk":
            st_short = "risk"
        elif pos_cat == "intervention":
            st_short = "interv"
        else:
            sub = attrs.get("subtype") or attrs.get("concept_category") or "body"
            st_short = SUBTYPE_SHORT.get(sub, sub[:2] if sub else "body")
        pieces.append(f"{name} [{st_short}]")
        if i < len(nodes) - 1:
            nxt = nodes[i + 1]
            ed = (
                edge_lookup.get((int(nid), int(nxt)))
                or edge_lookup.get((int(nxt), int(nid)))
                or {}
            )
            subtype = ed.get("subtype") or "EDGE"
            pieces.append(f" --[{subtype}]--> ")
    return "".join(pieces)


def assignment_id(side_decision):
    """side_decision is {existing: 'RG017'} or {new: {group_name, ...}}
    Return the resolved id string (or group_name) for indexing."""
    if not isinstance(side_decision, dict):
        return None
    if "existing" in side_decision:
        return side_decision["existing"]
    if "new" in side_decision:
        return side_decision["new"].get("group_name", "<unnamed>")
    return None


def load_merged_assignments_or_crash():
    """Read phase2_doublet_assignments.jsonl. CRASHES with named-artifact error if missing."""
    if not MERGED_JSONL_FP.exists():
        raise FileNotFoundError(
            f"\nERROR: required artifact missing.\n"
            f"  Expected: {MERGED_JSONL_FP}\n"
            f"  Produced by: `python phase2_step4_phase2_doublet_llm_grouping.py --mode seed|smoke|full`\n"
            f"               (after each batch completes, rebuild_merged_assignments_jsonl() writes this file)\n"
            f"  This script does NOT fall back to phase2_doublet_review.json or to "
            f"manual-baseline mode; run the upstream step first.\n"
        )
    rows = []
    with open(MERGED_JSONL_FP, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))  # let JSONDecodeError surface — no swallow
    if not rows:
        raise ValueError(
            f"ERROR: {MERGED_JSONL_FP} exists but contains zero rows. "
            f"Upstream grouping step produced no assignments. Investigate before re-running this script."
        )
    return rows


def load_catalog_or_crash():
    """Read seed_catalog.json AND active_catalog.json if it exists.
    CRASHES if seed_catalog.json missing."""
    if not CATALOG_FP.exists():
        raise FileNotFoundError(
            f"\nERROR: required artifact missing.\n"
            f"  Expected: {CATALOG_FP}\n"
            f"  Produced by: `python phase2_step4_phase2_doublet_llm_grouping.py --mode seed`\n"
            f"  This script does NOT fall back to per-subtype catalogs.\n"
        )
    catalog = json.loads(CATALOG_FP.read_text(encoding="utf-8"))
    if ACTIVE_CATALOG_FP.exists():
        active = json.loads(ACTIVE_CATALOG_FP.read_text(encoding="utf-8"))
        catalog["risk_groups"] = active["risk_groups"]
        catalog["mechanism_groups"] = active["mechanism_groups"]
        catalog["_catalog_source"] = "active"
    else:
        catalog["_catalog_source"] = "seed"
    return catalog


def build_xlsx_from_disc(
    paths_idx=None, node_attrs=None, edge_lookup=None, out_path=None
):
    """Build the spreadsheet from on-disc state. Can be called standalone (loads
    paths/node_attrs/edge_data from disc) OR from run_full() with pre-loaded data.
    out_path: if provided, write to this path instead of the default OUT_FP. Used
              by opus_review so Sonnet-era xlsx (`phase2_doublet_review_combined_v2.xlsx`)
              stays intact while post-review xlsx variants are written alongside it.
    """
    target_fp = Path(out_path) if out_path else OUT_FP
    catalog = load_catalog_or_crash()
    print(
        f"catalog source: {catalog['_catalog_source']} — "
        f"{len(catalog['risk_groups'])} RG + {len(catalog['mechanism_groups'])} MG"
    )
    merged = load_merged_assignments_or_crash()
    print(f"merged assignments: {len(merged)} rows")

    if paths_idx is None:
        print("loading paths ...")
        paths_idx = load_paths_indexed()
        print(f"  {len(paths_idx)} paths")
    if node_attrs is None:
        print("loading node attrs ...")
        with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
            node_attrs = pickle.load(f)
        print(f"  {len(node_attrs)} nodes")
    if edge_lookup is None:
        print("loading edge data + building lookup ...")
        with open(STEP1 / "graph_edge_data.pkl", "rb") as f:
            edge_data = pickle.load(f)
        edge_lookup = build_edge_lookup(edge_data)
        print(f"  {len(edge_lookup)} EDGE-type edges indexed")

    rg_members = defaultdict(list)
    mg_members = defaultdict(list)
    path_rows = []
    source_counts = defaultdict(int)

    # path_id -> latest record (later batches override seed if same path re-assigned)
    latest = {}
    for r in merged:
        pid = r.get("path_id")
        if pid:
            latest[pid] = r
            source_counts[r.get("source", "?")] += 1
    for pid, r in latest.items():
        rg = r.get("risk_group_id")
        mg = r.get("mechanism_group_id")
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
                "risk_group": rg or "",
                "mechanism_group": mg or "",
                "source": r.get("source", ""),
                "traversal": traversal,
                "notes": "",
            }
        )

    # Risk-groups sheet — LONG FORMAT: one row per (group, member_path). Group fields
    # repeat on each row for natural filter/sort by group_id. Empty groups get one
    # row with blank path_id/traversal so the full catalog is always visible.
    rg_rows = []
    for g in catalog.get("risk_groups", []):
        gid = g["group_id"]
        member_pids = rg_members.get(gid, [])
        if not member_pids:
            rg_rows.append(
                {
                    "group_id": gid,
                    "group_name": g["group_name"],
                    "description": g["group_description"],
                    "n_paths_total": 0,
                    "path_id": "",
                    "path_traversal": "",
                }
            )
        else:
            for pid in member_pids:
                p = paths_idx.get(pid)
                traversal = (
                    render_traversal(p, node_attrs, edge_lookup)
                    if p
                    else "(path not found)"
                )
                rg_rows.append(
                    {
                        "group_id": gid,
                        "group_name": g["group_name"],
                        "description": g["group_description"],
                        "n_paths_total": len(member_pids),
                        "path_id": pid,
                        "path_traversal": traversal,
                    }
                )
    # Mechanism-groups sheet — same long format
    mg_rows = []
    for g in catalog.get("mechanism_groups", []):
        gid = g["group_id"]
        member_pids = mg_members.get(gid, [])
        if not member_pids:
            mg_rows.append(
                {
                    "group_id": gid,
                    "group_name": g["group_name"],
                    "description": g["group_description"],
                    "n_paths_total": 0,
                    "path_id": "",
                    "path_traversal": "",
                }
            )
        else:
            for pid in member_pids:
                p = paths_idx.get(pid)
                traversal = (
                    render_traversal(p, node_attrs, edge_lookup)
                    if p
                    else "(path not found)"
                )
                mg_rows.append(
                    {
                        "group_id": gid,
                        "group_name": g["group_name"],
                        "description": g["group_description"],
                        "n_paths_total": len(member_pids),
                        "path_id": pid,
                        "path_traversal": traversal,
                    }
                )

    df_rg = pd.DataFrame(rg_rows)
    df_mg = pd.DataFrame(mg_rows)
    df_paths = pd.DataFrame(path_rows)

    summary_rows = [
        {
            "pool": "risk",
            "n_groups": len(catalog.get("risk_groups", [])),
            "n_assigned_paths": sum(len(v) for v in rg_members.values()),
        },
        {
            "pool": "mechanism",
            "n_groups": len(catalog.get("mechanism_groups", [])),
            "n_assigned_paths": sum(len(v) for v in mg_members.values()),
        },
    ]
    for src, n in sorted(source_counts.items()):
        summary_rows.append(
            {"pool": f"source:{src}", "n_groups": "", "n_assigned_paths": n}
        )
    df_summary = pd.DataFrame(summary_rows)

    print(f"\nwriting {target_fp.name} ...")
    with pd.ExcelWriter(target_fp, engine="openpyxl") as xw:
        df_summary.to_excel(xw, sheet_name="summary", index=False)
        df_rg.to_excel(xw, sheet_name="risk_groups", index=False)
        df_mg.to_excel(xw, sheet_name="mechanism_groups", index=False)
        df_paths.to_excel(xw, sheet_name="paths", index=False)

        # Column widths
        for sheet_name, df in [
            ("summary", df_summary),
            ("risk_groups", df_rg),
            ("mechanism_groups", df_mg),
            ("paths", df_paths),
        ]:
            ws = xw.sheets[sheet_name]
            for col_idx, col in enumerate(df.columns, start=1):
                if col in ("path_traversal", "traversal"):
                    width = 120
                elif col == "description":
                    width = 60
                elif col == "group_name":
                    width = 45
                else:
                    width = max(
                        12,
                        min(
                            40,
                            max(
                                [len(str(col))]
                                + [len(str(v)) if v else 0 for v in df[col].head(50)]
                            ),
                        ),
                    )
                ws.column_dimensions[
                    ws.cell(row=1, column=col_idx).column_letter
                ].width = width
            from openpyxl.styles import Alignment

            wrap_cols = {
                col: i + 1
                for i, col in enumerate(df.columns)
                if col in ("path_traversal", "traversal", "description", "group_name")
            }
            for row_idx in range(2, len(df) + 2):
                for col_name, col_letter_idx in wrap_cols.items():
                    cell = ws.cell(row=row_idx, column=col_letter_idx)
                    cell.alignment = Alignment(wrap_text=True, vertical="top")

    print(f"wrote {target_fp}")
    print(f"  summary:           {len(df_summary)} rows")
    print(f"  risk_groups:       {len(df_rg)} rows")
    print(f"  mechanism_groups:  {len(df_mg)} rows")
    print(f"  paths:             {len(df_paths)} rows")
    return {
        "out_path": str(target_fp),
        "n_risk_groups": len(df_rg),
        "n_mechanism_groups": len(df_mg),
        "n_paths": len(df_paths),
        "source_counts": dict(source_counts),
    }


if __name__ == "__main__":
    build_xlsx_from_disc()
