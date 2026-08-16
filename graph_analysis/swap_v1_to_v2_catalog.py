"""swap_v1_to_v2_catalog.py — Replace the active routing catalog + assignments
with the pilot v2 (new-architecture) outputs.

Actions:
  1. Backup v1 catalog -> phase2_routing_active_catalog.json.pre_v2_swap.bak
  2. Backup v1 assignments -> phase2_routing_assignments.jsonl.pre_v2_swap.bak
  3. Build new active catalog from v2 pilot output:
       - harm_classes from v2 (14)
       - mechanism_classes from v2 (19)
       - axes from v2 (6, with axis values; intervention/risk kinds preserved)
  4. Replace assignments jsonl with v2's flattened rows (includes fit_score,
     fit_note, harm_target_evidence, unassigned status, axes)
  5. Mark routing state to indicate v2 swap point
  6. Regenerate heuristic misfits + xlsx for v2 active state

Class B (no LLM). Idempotent.
"""

from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M
import phase2_step5_opus_routing as R

V1_CATALOG_FP = R.ACTIVE_CATALOG_FP
V1_ASG_FP = R.ASSIGNMENTS_FP
V1_STATE_FP = R.STATE_FP

V2_PILOT_FP = R.STEP1 / "phase2_pilot_v2_100paths_discovery.json"
V2_FLAT_ASG = R.STEP1 / "phase2_pilot_v2_100paths_assignments.jsonl"

BACKUP_CAT = R.STEP1 / "phase2_routing_active_catalog.json.pre_v2_swap.bak"
BACKUP_ASG = R.STEP1 / "phase2_routing_assignments.jsonl.pre_v2_swap.bak"
BACKUP_STATE = R.STEP1 / "phase2_routing_state.json.pre_v2_swap.bak"


def main():
    if not V2_PILOT_FP.exists():
        sys.exit(f"ERROR: v2 pilot missing at {V2_PILOT_FP}")
    if not V2_FLAT_ASG.exists():
        sys.exit(
            f"ERROR: v2 flattened assignments missing at {V2_FLAT_ASG}\n"
            f"  Run: python flatten_pilot_v2_to_jsonl.py"
        )

    # Backup v1
    if V1_CATALOG_FP.exists() and not BACKUP_CAT.exists():
        shutil.copy2(V1_CATALOG_FP, BACKUP_CAT)
        print(f"backed up v1 catalog -> {BACKUP_CAT.name}")
    if V1_ASG_FP.exists() and not BACKUP_ASG.exists():
        shutil.copy2(V1_ASG_FP, BACKUP_ASG)
        print(f"backed up v1 assignments -> {BACKUP_ASG.name}")
    if V1_STATE_FP.exists() and not BACKUP_STATE.exists():
        shutil.copy2(V1_STATE_FP, BACKUP_STATE)
        print(f"backed up v1 state -> {BACKUP_STATE.name}")

    # Build v2 catalog
    v2 = json.loads(V2_PILOT_FP.read_text(encoding="utf-8"))
    raw = v2["raw_output"]
    new_catalog = {
        "version": "v2_swapped_2026_05_18",
        "swapped_from_v1": True,
        "source_pilot": V2_PILOT_FP.name,
        "harm_classes": raw["harm_classes"],
        "mechanism_classes": raw["mechanism_classes"],
        "axes": raw["axes"],
        "group_remap": {},  # fresh remap; v1 IDs no longer apply
        "v1_history_note": "v1 catalog backed up at "
        f"{BACKUP_CAT.name}; v1 had 33 HC + 45 MC + 6 axes "
        "with many singletons. Swap 2026-05-18 after pilot v2 "
        "produced 14 HC + 19 MC + 6 axes with 0 singletons "
        "via new architecture (fit_score, unassigned, "
        "risk/intervention decoupling, paper-goal framing).",
    }
    M.atomic_write_json(V1_CATALOG_FP, new_catalog)
    print(
        f"wrote new active catalog: {len(new_catalog['harm_classes'])} HC + "
        f"{len(new_catalog['mechanism_classes'])} MC + "
        f"{len(new_catalog['axes'])} axes"
    )

    # Replace assignments jsonl with v2 flat
    shutil.copy2(V2_FLAT_ASG, V1_ASG_FP)
    print("copied v2 assignments -> active jsonl")

    # Reset state — v2 is the new baseline. Pilot is "already seeded".
    new_state = {
        "v2_swap_date": "2026-05-18",
        "total_batches_run": 0,
        "last_consolidation_at_batch_idx": -1,
        "total_consolidations_run": 0,
        "total_misfit_reviews_run": 0,
        "total_axes_reviews_run": 0,
        "total_final_sweeps_run": 0,
        "pilot_assignments_seeded": True,
        "active_pilot_source": V2_PILOT_FP.name,
    }
    M.atomic_write_json(V1_STATE_FP, new_state)
    print("reset routing state for v2 baseline")

    # Regenerate heuristic misfits (re-runs on v2 data)
    print("regenerating heuristic misfits ...")
    import subprocess

    res = subprocess.run(
        [sys.executable, "audit_pilot_v2_misfits.py"], capture_output=True, text=True
    )
    print(res.stdout)
    if res.returncode != 0:
        print(f"WARNING: audit script returned {res.returncode}: {res.stderr}")

    # Regenerate xlsx
    print("regenerating xlsx ...")
    res = subprocess.run(
        [sys.executable, "phase2_routing_to_xlsx.py"], capture_output=True, text=True
    )
    print(res.stdout)
    if res.returncode != 0:
        print(f"WARNING: xlsx script returned {res.returncode}: {res.stderr}")

    print("\n=== Swap complete ===")
    print(
        f"  active catalog: {len(new_catalog['harm_classes'])} HC + "
        f"{len(new_catalog['mechanism_classes'])} MC"
    )
    print(f"  active assignments: {V1_ASG_FP.name}")
    print("  state reset; pilot_assignments_seeded=True")
    print("  Next: run misfit_review on v2 for calibration, then routing batches.")


if __name__ == "__main__":
    main()
