"""calibration_rerun_100paths.py — Re-route the original 100 pilot paths through
the new routing prompt (active catalog + watch_items + fit_score + unassigned +
generalize/reassign flags + risk/intervention decoupling instruction) and write
to a separate namespace so the existing pilot data is preserved.

Comparison target: phase2_routing_quality_audit metrics
  pilot_v1_post_edits (old pilot, post catalog edits) vs calibration_100paths (this run)

Outputs:
  phase2_calibration_routing_100paths.json       - raw Opus output + parsed
  phase2_calibration_routing_assignments.jsonl   - resolved assignments
  phase2_calibration_routing_partial.txt         - streaming partial

DOES NOT touch:
  phase2_pilot_100paths_axis_discovery.json (original pilot)
  phase2_routing_active_catalog.json (active catalog — read-only)
  phase2_routing_assignments.jsonl (production routing assignments)
  phase2_routing_batches/ (production routing batches)

Class A. Estimated ~12-15pp Opus (100 paths via routing prompt with full
context, vs ~10pp for original pilot's combined seed-gen+routing prompt).
"""

from __future__ import annotations
import json
import pickle
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M
import phase2_step5_opus_routing as R

CALIB_OUT_FP = R.STEP1 / "phase2_calibration_routing_100paths.json"
CALIB_ASG_FP = R.STEP1 / "phase2_calibration_routing_assignments.jsonl"
CALIB_PARTIAL_FP = R.STEP1 / "phase2_calibration_routing_partial.txt"


def main():
    if CALIB_OUT_FP.exists():
        print(
            f"[idempotent skip] {CALIB_OUT_FP.name} exists. Delete to re-run.",
            flush=True,
        )
        return

    # Load original pilot to get the exact 100 input path_ids
    pilot = json.loads(R.PILOT_FP.read_text(encoding="utf-8"))
    target_pids = set(pilot.get("input_path_ids", []))
    if not target_pids:
        target_pids = {a["path_id"] for a in pilot["raw_output"]["assignments"]}
    print(f"target path_ids from pilot: {len(target_pids)}", flush=True)

    # Load deduped paths + filter to the 100 pilot paths
    print("loading deduped paths ...", flush=True)
    all_paths = R._load_deduped_paths()
    sample = [p for p in all_paths if p["path_id"] in target_pids]
    print(f"  matched {len(sample)} / {len(target_pids)} pilot paths", flush=True)
    if len(sample) != len(target_pids):
        missing = target_pids - {p["path_id"] for p in sample}
        print(
            f"  WARNING: missing {len(missing)} pilot paths: {sorted(missing)[:5]}...",
            flush=True,
        )

    # Load active catalog + node_attrs
    catalog = R._load_active_catalog()
    print(
        f"  active catalog: {len(catalog['harm_classes'])} HC + "
        f"{len(catalog['mechanism_classes'])} MC + "
        f"{len(catalog.get('axes', []))} axes",
        flush=True,
    )
    with open(R.STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)

    # Counts from CURRENT routing assignments (which include the pilot + any
    # batch routing already done)
    hc_counts, mc_counts = R._compute_class_counts()
    print(
        f"  class counts populated: HC distinct={len(hc_counts)}, "
        f"MC distinct={len(mc_counts)}",
        flush=True,
    )

    # Build routing prompt (this is the SAME prompt main routing uses, just
    # with all 100 paths in one batch instead of 75)
    sentinel = uuid.uuid4().hex[:12]
    prompt = R.make_routing_prompt(sample, catalog, hc_counts, mc_counts, na, sentinel)
    print(f"prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)

    # Stream Opus call
    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        "calibration_100",
        CALIB_PARTIAL_FP,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(
            f"\nCALIBRATION FAILED ({err}). Partial preserved at {CALIB_PARTIAL_FP.name}",
            flush=True,
        )
        sys.exit(2)
    try:
        parsed = json.loads(json_part)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}", flush=True)
        last = json_part.rfind('{"assignments":[')
        if last > 0:
            try:
                parsed = json.loads(json_part[last:])
                print(
                    f"  RECOVERED via rfind-restart (dropped {last} chars)", flush=True
                )
            except Exception as e2:
                print(f"  RECOVERY FAILED: {e2}", flush=True)
                sys.exit(3)
        else:
            sys.exit(3)

    # Apply resolver (MIN_GROUP_SIZE enforcement etc.) BUT DO NOT mutate
    # the production active catalog — pass a deep copy.
    cat_copy = json.loads(json.dumps(catalog))
    cat_after, resolved, forced, dropped_hc, dropped_mc = R._resolve_and_enforce(
        parsed, cat_copy, batch_idx=-1
    )  # -1 marks calibration batch

    # Stamp source so quality audit identifies these clearly
    for r in resolved:
        r["source"] = "calibration_100paths"

    M.atomic_write_json(
        CALIB_OUT_FP,
        {
            "calibration_n_paths": len(sample),
            "duration_sec": dur,
            "n_input": len(sample),
            "n_resolved": len(resolved),
            "n_forced_fits": forced,
            "dropped_new_hc_names": dropped_hc,
            "dropped_new_mc_names": dropped_mc,
            "n_hc_proposed_new": sum(
                1
                for a in parsed.get("assignments", [])
                if isinstance(a.get("harm_class"), dict)
                and "new" in a.get("harm_class", {})
            ),
            "n_mc_proposed_new": sum(
                1
                for a in parsed.get("assignments", [])
                if isinstance(a.get("mechanism_class"), dict)
                and "new" in a.get("mechanism_class", {})
            ),
            "n_hc_unassigned": sum(
                1 for r in resolved if r.get("harm_class_status") == "unassigned"
            ),
            "n_mc_unassigned": sum(
                1 for r in resolved if r.get("mechanism_class_status") == "unassigned"
            ),
            "catalog_flags": parsed.get("catalog_flags", {}),
            "raw_output": parsed,
            "resolved_assignments": resolved,
        },
    )
    with open(CALIB_ASG_FP, "w", encoding="utf-8") as f:
        for r in resolved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {CALIB_OUT_FP.name}", flush=True)
    print(f"  resolved: {len(resolved)}", flush=True)
    print(f"  forced-fits: {forced}", flush=True)
    print(f"  HC dropped sub-min: {dropped_hc}", flush=True)
    print(f"  MC dropped sub-min: {dropped_mc}", flush=True)
    print(
        f"  HC unassigned: {sum(1 for r in resolved if r.get('harm_class_status') == 'unassigned')}",
        flush=True,
    )
    print(
        f"  MC unassigned: {sum(1 for r in resolved if r.get('mechanism_class_status') == 'unassigned')}",
        flush=True,
    )
    print("\nNext: run quality audit + diff vs pilot:", flush=True)
    print(
        f"  python phase2_routing_quality_audit.py --asg {CALIB_ASG_FP.name} --label calibration_100paths",
        flush=True,
    )
    print(
        "  python phase2_routing_quality_audit.py --compare pilot_v1_post_edits calibration_100paths",
        flush=True,
    )


if __name__ == "__main__":
    main()
