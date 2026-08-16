"""flatten_pilot_v2_to_jsonl.py — Convert pilot v2 JSON assignments to the
jsonl shape expected by phase2_routing_quality_audit.py.

Class B (no LLM). Idempotent.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M

PILOT_V2_FP = M.STEP1 / "phase2_pilot_v2_100paths_discovery.json"
OUT_FP = M.STEP1 / "phase2_pilot_v2_100paths_assignments.jsonl"


def main():
    d = json.loads(PILOT_V2_FP.read_text(encoding="utf-8"))
    rows = []
    for a in d["raw_output"]["assignments"]:
        hc = a.get("harm_class_id")
        mc = a.get("mechanism_class_id")
        hc_id = None
        hc_status = "assigned"
        if isinstance(hc, dict) and hc.get("unassigned"):
            hc_status = "unassigned"
        else:
            hc_id = hc
        mc_id = None
        mc_status = "assigned"
        if isinstance(mc, dict) and mc.get("unassigned"):
            mc_status = "unassigned"
        else:
            mc_id = mc
        rows.append(
            {
                "path_id": a["path_id"],
                "harm_class_id": hc_id,
                "harm_class_status": hc_status,
                "mechanism_class_id": mc_id,
                "mechanism_class_status": mc_status,
                "axes": a.get("axis_values", {}),
                "harm_target_evidence": a.get("harm_target_evidence"),
                "confidence": a.get("confidence"),
                "fit_score": a.get("fit_score"),
                "fit_note": a.get("fit_note"),
                "source": "pilot_v2_discovery",
            }
        )
    with open(OUT_FP, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_hc_unassigned = sum(1 for r in rows if r["harm_class_status"] == "unassigned")
    n_mc_unassigned = sum(
        1 for r in rows if r["mechanism_class_status"] == "unassigned"
    )
    print(f"wrote {OUT_FP.name}: {len(rows)} rows", flush=True)
    print(f"  HC unassigned: {n_hc_unassigned}", flush=True)
    print(f"  MC unassigned: {n_mc_unassigned}", flush=True)


if __name__ == "__main__":
    main()
