"""recover_pilot_v2_json.py — Recover the pilot v2 output from the partial file.

The Opus stream completed cleanly (60026 chars, SENTINEL OK) but the JSON had
invalid `\\'` escapes inside string values (JSON does not allow apostrophe
escaping with backslash; only `\\"` `\\\\` `\\/` `\\b` `\\f` `\\n` `\\r` `\\t` `\\uXXXX`).

This script replaces `\\'` -> `'` and parses, then writes the canonical pilot v2
output JSON.

Class B (no LLM tokens). Idempotent.
"""

from __future__ import annotations
import json
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M

PARTIAL_FP = M.STEP1 / "phase2_pilot_v2_100paths_discovery_partial.txt"
V1_FP = M.STEP1 / "phase2_pilot_100paths_axis_discovery.json"
OUT_FP = M.STEP1 / "phase2_pilot_v2_100paths_discovery.json"


def main():
    if OUT_FP.exists():
        print(f"[idempotent skip] {OUT_FP.name} exists.", flush=True)
        return
    raw = PARTIAL_FP.read_text(encoding="utf-8")
    sent_prefix = "END_SENTINEL_"
    i = raw.rfind(sent_prefix)
    if i < 0:
        sys.exit("ERROR: sentinel not found in partial")
    sentinel = raw[i + len(sent_prefix) :].strip()
    json_part = raw[:i]
    print(f"sentinel: {sentinel}; json: {len(json_part)} chars", flush=True)

    # Known Opus issue: trailing architecture_critique string missing closing quote.
    # The JSON ends with `...publishable.}` but should be `...publishable."}`.
    fixed = json_part
    if fixed.endswith("}") and not fixed.endswith('"}'):
        fixed = fixed[:-1] + '"}'
        print('applied trailing-quote fix: ended with `}` -> `"}`', flush=True)

    parsed = None
    try:
        parsed = json.loads(fixed)
    except json.JSONDecodeError as e:
        print(f"first parse failed: {e}", flush=True)
        # Try also stripping invalid escapes
        fixed2 = re.sub(r'\\([^"\\/bfnrtu])', r"\1", fixed)
        try:
            parsed = json.loads(fixed2)
            print("  RECOVERED via invalid-escape strip", flush=True)
        except json.JSONDecodeError as e2:
            print(f"second parse failed: {e2}", flush=True)
            print(
                f"context: {fixed2[max(0, e2.pos - 100) : e2.pos + 100]!r}", flush=True
            )
            sys.exit(1)

    print("PARSED OK", flush=True)
    print(f"  axes:         {len(parsed.get('axes', []))}", flush=True)
    print(f"  harm_classes: {len(parsed.get('harm_classes', []))}", flush=True)
    print(f"  mech_classes: {len(parsed.get('mechanism_classes', []))}", flush=True)
    print(f"  assignments:  {len(parsed.get('assignments', []))}", flush=True)

    v1 = json.loads(V1_FP.read_text(encoding="utf-8"))
    M.atomic_write_json(
        OUT_FP,
        {
            "pilot_n": 100,
            "duration_sec": 1044,
            "version": "v2_discovery_new_architecture",
            "recovered_from_partial": True,
            "recovery_method": "manual: replace invalid JSON escapes (\\' etc) in partial.txt",
            "v1_path_ids_source": V1_FP.name,
            "n_axes": len(parsed.get("axes", [])),
            "n_harm_classes": len(parsed.get("harm_classes", [])),
            "n_mech_classes": len(parsed.get("mechanism_classes", [])),
            "n_assignments": len(parsed.get("assignments", [])),
            "raw_output": parsed,
            "input_path_ids": v1["input_path_ids"],
        },
    )
    print(f"\nwrote {OUT_FP.name}", flush=True)


if __name__ == "__main__":
    main()
