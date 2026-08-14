"""One-off recovery for HC013 sweep_018 chunk-2 BrokenPipe failure.

Streams Opus on the 59 untouched path_ids (computed via set-diff: current_HC013 -
sweep_018_c00 decided), seeded with sweep_018 chunk 1's 4 SPLIT_OUT proposals as
prior_chunk_proposals. Writes the result as sweep_019_HC_HC013_c00.json so the
main script's idempotent-skip path can apply the decisions on next invocation:

    python -u dive_hc013_chunk2_recovery.py
    python -u phase2_step5_opus_routing.py --mode final_misfit_sweep \\
        --classes HC013 --chunk-size 200
"""

import json
import pickle
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import phase2_step4_phase2_doublet_llm_grouping as M  # noqa: E402
from phase2_step5_opus_routing import (  # noqa: E402
    STEP1,
    ASSIGNMENTS_FP,
    _load_active_catalog,
    _load_deduped_paths,
    _partial_path,
    make_class_sweep_prompt,
)

CLASS_ID = "HC013"
CLASS_KIND = "HC"
SWEEP_IDX = 19  # next sweep_idx after 018
CHUNK_IDX = 0
RECOVERY_INPUT = STEP1 / "sweep_019_HC_HC013_recovery_input.jsonl"
OUT_DIR = STEP1 / "phase2_routing_final_misfit_sweep"
OUT_FP = (
    OUT_DIR / f"sweep_{SWEEP_IDX:03d}_{CLASS_KIND}_{CLASS_ID}_c{CHUNK_IDX:02d}.json"
)


def main():
    print(f"=== HC013 chunk-2 recovery (writes {OUT_FP.name}) ===", flush=True)

    if OUT_FP.exists():
        print(
            f"REFUSING: {OUT_FP} already exists. Delete it manually if intended re-run.",
            flush=True,
        )
        sys.exit(1)

    # Load recovery input (header + 59 path_ids)
    lines = [
        json.loads(line)
        for line in RECOVERY_INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    header = lines[0]
    assert header.get("_type") == "recovery_header", (
        "first line must be recovery_header"
    )
    target_path_ids = [r["path_id"] for r in lines[1:]]
    prior_chunk_proposals = header["prior_chunk_proposals"]
    print(
        f"  loaded recovery input: {len(target_path_ids)} target paths, "
        f"{len(prior_chunk_proposals)} prior_chunk_proposals",
        flush=True,
    )
    for p in prior_chunk_proposals:
        print(f"    [{p['member_count']} prior verdicts] {p['name']}", flush=True)

    # Load catalog + assignments + paths + node_attrs
    catalog = _load_active_catalog()
    rows = [
        json.loads(line)
        for line in ASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paths = _load_deduped_paths()
    paths_idx = {p["path_id"]: p for p in paths}
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    print(
        f"  loaded catalog ({len(catalog['harm_classes'])} HC + "
        f"{len(catalog['mechanism_classes'])} MC), "
        f"{len(rows)} assignment rows, {len(paths)} paths",
        flush=True,
    )

    # Find HC013 entry
    class_entry = next(
        (h for h in catalog["harm_classes"] if h["class_id"] == CLASS_ID), None
    )
    assert class_entry is not None, f"{CLASS_ID} not in catalog"

    # Filter assignment rows to the 59 target paths (must all be currently in HC013)
    target_set = set(target_path_ids)
    member_rows = [
        r
        for r in rows
        if r["path_id"] in target_set and r.get("harm_class_id") == CLASS_ID
    ]
    missing = target_set - {r["path_id"] for r in member_rows}
    if missing:
        print(
            f"WARN: {len(missing)} target path_ids not found in HC013 assignments — "
            f"they may have been moved since recovery input was written:",
            flush=True,
        )
        for m in sorted(missing)[:10]:
            print(f"    {m}", flush=True)
    print(f"  resolved {len(member_rows)} member_rows for streaming", flush=True)

    # Render MC listing (cross-side context for HC sweep)
    mc_listing = "\n".join(
        f"  {m['class_id']}: {m['class_name']} — "
        f"{(m.get('class_description') or '')[:120]}"
        for m in catalog["mechanism_classes"]
    )
    other_str = "MECHANISM CLASSES:\n" + mc_listing

    # Build prompt with prior_chunk_proposals seeded
    sentinel = uuid.uuid4().hex[:12]
    prompt = make_class_sweep_prompt(
        catalog,
        CLASS_KIND,
        class_entry,
        member_rows,
        other_str,
        na,
        paths_idx,
        sentinel,
        prior_chunk_proposals=prior_chunk_proposals,
    )
    print(
        f"  built prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True
    )

    label = f"sweep_{SWEEP_IDX:03d}_{CLASS_ID}_c{CHUNK_IDX:02d}_recovery"
    partial = _partial_path(label)
    print(f"  partial file: {partial}", flush=True)

    # Stream
    t0 = time.time()
    json_part, dur, attempt, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        label,
        partial,
    )
    if err or json_part is None:
        print(f"FAILED: {err}", flush=True)
        sys.exit(2)

    parsed = json.loads(json_part)
    n_dec = len(parsed.get("decisions", []))
    print(f"  parsed {n_dec} decisions from stream (took {dur:.0f}s)", flush=True)

    # Write in the SAME schema the main script's idempotent skip expects
    out_payload = {
        "sweep_idx": SWEEP_IDX,
        "class_kind": CLASS_KIND,
        "class_id": CLASS_ID,
        "chunk_idx": CHUNK_IDX,
        "n_chunks": 1,  # recovery treats this as a single-chunk call
        "n_in_chunk": len(member_rows),
        "duration_sec": dur,
        "raw_output": parsed,
        "_recovery_note": "manual recovery of sweep_018 chunk 2 (BrokenPipe failure); "
        "seeded with sweep_018 c00's 4 SPLIT_OUT proposals as prior_chunk_proposals",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FP.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"  wrote {OUT_FP}", flush=True)

    # Verdict summary
    from collections import Counter

    vc = Counter(d.get("verdict") for d in parsed.get("decisions", []))
    print(f"  verdict counts: {dict(vc)}", flush=True)

    print(f"=== Recovery complete in {time.time() - t0:.0f}s ===", flush=True)
    print(
        f"Next step: python -u phase2_step5_opus_routing.py --mode final_misfit_sweep "
        f"--classes {CLASS_ID} --chunk-size 200",
        flush=True,
    )


if __name__ == "__main__":
    main()
