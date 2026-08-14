"""
phase2_step4_phase2_review_to_xlsx.py — Convert combined review JSONs to one XLSX
spreadsheet with all subtypes listed.

Output: phase2_full_vpn_review_combined_nr/combined_review_all_subtypes.xlsx
  Sheet "by_group"     — one row per (subtype, group). Compact summary with all member names concatenated.
  Sheet "by_member"    — one row per (subtype, group, member). Detailed for filtering.
  Sheet "summary"      — per-subtype totals.
  Sheet "no_fit"       — pass-A review no_fit + smoke residual nodes (per subtype).
"""

import json
import sys
from pathlib import Path

import pandas as pd

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
# Canonical logical-chain order (per project CLAUDE.md): risk → pa → ti → dr → im → va → intervention
# risk is the chain head; NR body+intervention follow
SUBTYPE_ORDER = [
    "risk",
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
    "intervention",
]
# pool lookup so each subtype reads from the right combined dir
SUBTYPE_POOL = {st: ("risk" if st == "risk" else "nr") for st in SUBTYPE_ORDER}


def main():
    by_group_rows = []
    by_member_rows = []
    no_fit_rows = []
    summary_rows = []

    for st in SUBTYPE_ORDER:
        st_short = SUBTYPE_SHORT[st]
        pool = SUBTYPE_POOL[st]
        fp = (
            STEP1
            / f"phase2_full_vpn_review_combined_{pool}"
            / f"combined_{st_short}.json"
        )
        if not fp.exists():
            print(f"SKIP {st} (pool={pool}): {fp} missing")
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))

        # Summary row
        summary_rows.append(
            {
                "pool": pool,
                "subtype": st,
                "n_groups_total": d["n_groups_total"],
                "n_groups_with_members": d["n_groups_with_members"],
                "n_groups_empty": d["n_groups_empty"],
                "n_members_from_pass_a_review": d["n_members_from_pass_a_review"],
                "n_members_from_smoke_batch": d["n_members_from_smoke_batch"],
                "n_no_fit_pass_a_review": d["n_no_fit_pass_a_review"],
                "n_residual_smoke": d["n_residual_smoke"],
            }
        )

        for g in d["groups"]:
            member_names = [m.get("name", "") for m in g["members"]]
            by_group_rows.append(
                {
                    "pool": pool,
                    "subtype": st,
                    "group_name": g["group_name"],
                    "description": g["description"],
                    "n_distinct_members": g["n_distinct_members"],
                    "n_only_review": g["n_only_review"],
                    "n_only_smoke": g["n_only_smoke"],
                    "n_both": g["n_both_sources_same_decision"],
                    "member_names_concat": " | ".join(member_names),
                }
            )
            for m in g["members"]:
                by_member_rows.append(
                    {
                        "pool": pool,
                        "subtype": st,
                        "group_name": g["group_name"],
                        "group_description": g["description"],
                        "node_id": m.get("node_id"),
                        "node_name": m.get("name", ""),
                        "sources": "+".join(m.get("sources", [])),
                        "smoke_decision": m.get("smoke_decision", ""),
                        "smoke_confidence": m.get("smoke_confidence", ""),
                    }
                )

        for m in d.get("no_fit_pass_a_review_nodes", []):
            no_fit_rows.append(
                {
                    "pool": pool,
                    "subtype": st,
                    "kind": "pass_a_review_no_fit",
                    "node_id": m.get("node_id"),
                    "node_name": m.get("name", ""),
                }
            )
        for m in d.get("residual_smoke_nodes", []):
            no_fit_rows.append(
                {
                    "pool": pool,
                    "subtype": st,
                    "kind": "smoke_residual",
                    "node_id": m.get("node_id"),
                    "node_name": m.get("name", ""),
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    df_by_group = pd.DataFrame(by_group_rows)
    df_by_member = pd.DataFrame(by_member_rows)
    df_no_fit = pd.DataFrame(no_fit_rows)

    # Write to NR combined dir (contains union of NR+risk) — single workbook
    out_dir = STEP1 / "phase2_full_vpn_review_combined_nr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_review_all_subtypes.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_summary.to_excel(xw, sheet_name="summary", index=False)
        df_by_group.to_excel(xw, sheet_name="by_group", index=False)
        df_by_member.to_excel(xw, sheet_name="by_member", index=False)
        df_no_fit.to_excel(xw, sheet_name="no_fit", index=False)

        # Set sensible column widths
        for sheet_name, df in [
            ("summary", df_summary),
            ("by_group", df_by_group),
            ("by_member", df_by_member),
            ("no_fit", df_no_fit),
        ]:
            ws = xw.sheets[sheet_name]
            for col_idx, col in enumerate(df.columns, start=1):
                max_len = max(
                    [len(str(col))]
                    + [
                        min(len(str(v)), 120) if v is not None else 0
                        for v in df[col].head(200)
                    ]
                )
                ws.column_dimensions[
                    ws.cell(row=1, column=col_idx).column_letter
                ].width = min(max_len + 2, 80)

    print(f"wrote {out_path}")
    print(f"  summary:   {len(df_summary)} rows")
    print(f"  by_group:  {len(df_by_group)} rows")
    print(f"  by_member: {len(df_by_member)} rows")
    print(f"  no_fit:    {len(df_no_fit)} rows")


if __name__ == "__main__":
    main()
