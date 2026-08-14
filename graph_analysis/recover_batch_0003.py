"""recover_batch_0003.py — One-shot recovery of batch_0003 from the failed
partial. Opus dropped the trailing outer-object closing brace; appending "}"
yields valid JSON. This script:
  1. Reads the preserved partial
  2. Strips sentinel + appends missing "}"
  3. Calls _resolve_and_enforce + _append_flags + writes batch_0003.json
  4. Rebuilds the merged assignments jsonl
  5. Bumps state.total_batches_run by 1
  6. Regenerates xlsx
Class B (no LLM tokens).
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M
import phase2_step5_opus_routing as R

PARTIAL_FP = R.STEP1 / "phase2_routing_partial.batch_0003_failed.txt"
BATCH_IDX = 3


def main():
    if not PARTIAL_FP.exists():
        sys.exit(f"ERROR: {PARTIAL_FP} missing")
    raw = PARTIAL_FP.read_text(encoding="utf-8")
    m = re.search(r"END_SENTINEL_[a-f0-9]+", raw)
    if not m:
        sys.exit("ERROR: no END_SENTINEL found in partial")
    payload = raw[: m.start()]
    # Trailing-brace fix
    fixed = payload + "}"
    try:
        parsed = json.loads(fixed)
    except json.JSONDecodeError as e:
        sys.exit(f"recovery failed: {e}")
    print(
        f"recovered: {len(parsed.get('assignments', []))} assignments + "
        f"{len(parsed.get('catalog_flags', {}))} flag categories"
    )

    catalog = R._load_active_catalog()
    catalog, resolved, forced, dropped_hc, dropped_mc = R._resolve_and_enforce(
        parsed, catalog, BATCH_IDX
    )
    R._append_flags(parsed, BATCH_IDX)
    R._save_active_catalog(
        catalog, kind_suffix=f"routing_batch_{BATCH_IDX:04d}_recovered"
    )

    batch_out = R.BATCH_DIR / f"batch_{BATCH_IDX:04d}.json"
    M.atomic_write_json(
        batch_out,
        {
            "batch_idx": BATCH_IDX,
            "model": "claude-opus-4-7",
            "recovered_from": PARTIAL_FP.name,
            "recovery_method": "trailing-brace-append",
            "n_input_paths": len(parsed.get("assignments", [])),
            "duration_sec": 564.0,
            "assignments": parsed.get("assignments", []),
            "resolved_assignments": resolved,
            "catalog_flags": parsed.get("catalog_flags", {}),
            "dropped_new_hc_names": dropped_hc,
            "dropped_new_mc_names": dropped_mc,
            "forced_fit_count": forced,
        },
    )
    print(f"wrote {batch_out.name}")
    print(
        f"  resolved={len(resolved)} forced={forced} "
        f"dropped_hc={dropped_hc} dropped_mc={dropped_mc}"
    )

    n_rows = R._rebuild_routing_assignments_jsonl()
    print(f"merged jsonl now {n_rows} rows")

    # Bump state
    state = R._load_state()
    state["total_batches_run"] = state.get("total_batches_run", 0) + 1
    R._save_state(state)
    print(f"state.total_batches_run -> {state['total_batches_run']}")

    # Regenerate xlsx
    try:
        from phase2_routing_to_xlsx import build_routing_xlsx

        res = build_routing_xlsx()
        print(
            f"xlsx refreshed: {res['n_paths']} paths, {res['n_harm_classes']} HC + "
            f"{res['n_mech_classes']} MC"
        )
    except Exception as e:
        print(f"xlsx rebuild error: {e}")


if __name__ == "__main__":
    main()
