"""study_b_sonnet_context_sensitivity.py

Sensitivity test: does adding PAPER_DELIVERABLE_CONTEXT to the Sonnet routing
prompt (make_assign_prompt) change assignment decisions vs the no-context
baseline?

Method: re-run the §19.13.3 30-path congruence test (Sonnet 4.6 on the smoke
catalog 20 RG + 27 MG) WITH the new context prepended. Compare:
  (a) New WITH-context Sonnet vs original NO-context Sonnet (sonnet_30path_congruence_report.json)
  (b) New WITH-context Sonnet vs original Opus seed assignments

If WITH-context Sonnet matches NO-context Sonnet >=27/30, the context doesn't
change Sonnet routing behavior in a meaningful way (safe to deploy or skip).
If WITH-context Sonnet matches Opus BETTER than NO-context did (29/30), then
context helps — worth deploying. Else, context is neutral.

Cost: 1 Sonnet call on 30 paths ~= 0.5pp session, 1-2 min wall.

Outputs:
  phase2_results/study_b_sonnet_with_context_assignments.json — raw output
  phase2_results/study_b_context_sensitivity_report.json — comparison
"""

from __future__ import annotations
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M

SMOKE_CATALOG = (
    M.STEP1 / "archive" / "phase2_doublet_seed_catalog_v2_smoke_30paths.json"
)
BASELINE_REPORT = Path("logfiles/sonnet_30path_congruence_report.json")
OUT_RAW = M.STEP1.parent / "study_b_sonnet_with_context_assignments.json"
OUT_REPORT = M.STEP1.parent / "study_b_context_sensitivity_report.json"


def main():
    if not SMOKE_CATALOG.exists():
        print(f"ERROR: smoke catalog missing at {SMOKE_CATALOG}", flush=True)
        sys.exit(1)
    if not BASELINE_REPORT.exists():
        print(
            f"ERROR: baseline congruence report missing at {BASELINE_REPORT}",
            flush=True,
        )
        sys.exit(1)

    smoke = json.loads(SMOKE_CATALOG.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    print(
        f"smoke catalog: {smoke['n_risk_groups']} RG, {smoke['n_mechanism_groups']} MG",
        flush=True,
    )
    print(
        f"baseline (no-context Sonnet) results: both_match={baseline['both_match']}, "
        f"rg_match={baseline['rg_match']}, mg_match={baseline['mg_match']}",
        flush=True,
    )

    # Parse original Opus assignments from smoke (same logic as the baseline test)
    opus_by_pid = {}
    for a in smoke["assignments"]:
        pid = a["path_id"]
        rg = a.get("risk_group_id")
        mg = a.get("mechanism_group_id")
        opus_by_pid[pid] = {"rg": rg, "mg": mg}
    print(f"Opus original assignments: {len(opus_by_pid)} paths", flush=True)

    # Load paths
    paths, node_attrs = M.load_paths_and_attrs()
    smoke_pids = set(smoke["input_path_ids"])
    sample = [p for p in paths if p["path_id"] in smoke_pids]
    print(f"loaded {len(sample)} paths matching smoke set", flush=True)

    # Build prompt WITH paper-deliverable context prepended
    sentinel = uuid.uuid4().hex[:12]
    base_prompt = M.make_assign_prompt(
        sample,
        node_attrs,
        smoke["risk_groups"],
        smoke["mechanism_groups"],
        sentinel,
        allow_new=False,
        allow_coherence=False,
    )
    contextualized_prompt = M.PAPER_DELIVERABLE_CONTEXT + "\n\n" + base_prompt
    print(
        f"prompt: {len(contextualized_prompt)} chars "
        f"(~{len(contextualized_prompt) // 4} tokens) "
        f"[no-context baseline was ~47k tokens]",
        flush=True,
    )

    # Send to Sonnet
    partial = Path("logfiles/study_b_sonnet_with_context_partial.txt")
    print("\nlaunching Sonnet 4.6 with paper-context prefix ...", flush=True)
    text, dur, err = M.streaming_claude_call(
        contextualized_prompt,
        (
            "You produce STRICT JSON output for an AI-safety doublet grouping pipeline. "
            "Never preamble, never use markdown fences, always emit valid JSON, always "
            "end your output with the requested sentinel."
        ),
        partial,
        model="claude-sonnet-4-6",
    )
    print(f"sonnet returned: {len(text)} chars in {dur:.1f}s, err={err}", flush=True)
    if err:
        print(f"FAILED: {err}", flush=True)
        sys.exit(2)

    end_marker = f"END_SENTINEL_{sentinel}"
    trimmed = text.strip()
    if not (trimmed.startswith("{") and trimmed.endswith(end_marker)):
        print(
            f"FAIL sentinel validation; first/last 100: "
            f"{trimmed[:100]!r} ... {trimmed[-100:]!r}",
            flush=True,
        )
        # Try recovery
        sys.exit(3)
    json_part = trimmed[: -len(end_marker)].rstrip()
    try:
        parsed = json.loads(json_part)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}; trying rfind recovery", flush=True)
        last = json_part.rfind('{"assignments":[')
        if last > 0:
            try:
                parsed = json.loads(json_part[last:])
                print(f"  RECOVERED (dropped {last} chars)", flush=True)
            except Exception as e2:
                print(f"  RECOVERY FAILED: {e2}", flush=True)
                sys.exit(4)
        else:
            sys.exit(4)

    sonnet_w_ctx_by_pid = {}
    for a in parsed.get("assignments", []):
        pid = a["path_id"]
        rg = (a.get("risk_group") or {}).get("existing")
        mg = (a.get("mechanism_group") or {}).get("existing")
        sonnet_w_ctx_by_pid[pid] = {"rg": rg, "mg": mg}
    print(f"parsed: {len(sonnet_w_ctx_by_pid)} assignments", flush=True)

    OUT_RAW.write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  saved {OUT_RAW.name}", flush=True)

    # Re-extract original Sonnet (no-context) assignments from baseline report.
    # Baseline file has disagreements but not full per-pid; we reconstruct from
    # disagreements + assume rest match Opus (since both_match=29/30 means 29 paths
    # had Sonnet == Opus).
    # If baseline schema differs, fall back to reading the test_partial directly.
    baseline_partial = Path("logfiles/sonnet_30path_test_partial.txt")
    sonnet_no_ctx_by_pid = {}
    if baseline_partial.exists():
        bp_text = baseline_partial.read_text(encoding="utf-8")
        # Strip sentinel and parse
        for line in [bp_text]:
            # crude — assume the partial is the raw stream; find last assignments-start
            last = line.rfind('{"assignments":[')
            if last >= 0:
                try:
                    bp_json = line[last:]
                    # Remove trailing sentinel etc.
                    end_idx = bp_json.find("END_SENTINEL_")
                    if end_idx > 0:
                        bp_json = bp_json[:end_idx]
                    bp_json = bp_json.rstrip().rstrip(",")
                    # Try parsing as JSON; if it ends mid-object, find last complete
                    try:
                        bp_parsed = json.loads(bp_json)
                    except json.JSONDecodeError:
                        # Try removing final partial entry
                        bp_parsed = None
                    if bp_parsed:
                        for a in bp_parsed.get("assignments", []):
                            pid = a["path_id"]
                            rg = (a.get("risk_group") or {}).get("existing")
                            mg = (a.get("mechanism_group") or {}).get("existing")
                            sonnet_no_ctx_by_pid[pid] = {"rg": rg, "mg": mg}
                except Exception as e:
                    print(f"  WARN: couldn't reparse baseline partial: {e}", flush=True)

    # Compare: WITH-context vs Opus
    common_w_opus = set(sonnet_w_ctx_by_pid) & set(opus_by_pid)
    both_w_opus = sum(
        1 for pid in common_w_opus if sonnet_w_ctx_by_pid[pid] == opus_by_pid[pid]
    )
    rg_w_opus = sum(
        1
        for pid in common_w_opus
        if sonnet_w_ctx_by_pid[pid]["rg"] == opus_by_pid[pid]["rg"]
    )
    mg_w_opus = sum(
        1
        for pid in common_w_opus
        if sonnet_w_ctx_by_pid[pid]["mg"] == opus_by_pid[pid]["mg"]
    )

    # WITH-context vs NO-context Sonnet
    sonnet_self_compare = None
    if sonnet_no_ctx_by_pid:
        common_sonnets = set(sonnet_w_ctx_by_pid) & set(sonnet_no_ctx_by_pid)
        both_self = sum(
            1
            for pid in common_sonnets
            if sonnet_w_ctx_by_pid[pid] == sonnet_no_ctx_by_pid[pid]
        )
        rg_self = sum(
            1
            for pid in common_sonnets
            if sonnet_w_ctx_by_pid[pid]["rg"] == sonnet_no_ctx_by_pid[pid]["rg"]
        )
        mg_self = sum(
            1
            for pid in common_sonnets
            if sonnet_w_ctx_by_pid[pid]["mg"] == sonnet_no_ctx_by_pid[pid]["mg"]
        )
        sonnet_self_compare = {
            "n_common": len(common_sonnets),
            "both_match": both_self,
            "rg_match": rg_self,
            "mg_match": mg_self,
        }

    report = {
        "study": "Sonnet routing — paper-deliverable context sensitivity",
        "n_paths": len(sonnet_w_ctx_by_pid),
        "baseline_no_context_sonnet_vs_opus": {
            "both_match": baseline["both_match"],
            "rg_match": baseline["rg_match"],
            "mg_match": baseline["mg_match"],
        },
        "new_with_context_sonnet_vs_opus": {
            "n_common": len(common_w_opus),
            "both_match": both_w_opus,
            "rg_match": rg_w_opus,
            "mg_match": mg_w_opus,
        },
        "with_context_vs_no_context_sonnet": sonnet_self_compare,
        "wall_clock_sec_with_context": dur,
        "wall_clock_sec_no_context_baseline": baseline.get("sonnet_duration_sec"),
        "prompt_size_chars_with_context": len(contextualized_prompt),
    }
    OUT_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("STUDY B — Sonnet routing context sensitivity")
    print("=" * 70)
    print(
        f"  baseline no-context Sonnet vs Opus:  "
        f"both={baseline['both_match']}, rg={baseline['rg_match']}, mg={baseline['mg_match']}"
    )
    print(
        f"  new with-context Sonnet vs Opus:     "
        f"both={both_w_opus}, rg={rg_w_opus}, mg={mg_w_opus}"
    )
    if sonnet_self_compare:
        print(
            f"  with-context vs no-context Sonnet:   "
            f"both={sonnet_self_compare['both_match']}, "
            f"rg={sonnet_self_compare['rg_match']}, "
            f"mg={sonnet_self_compare['mg_match']} "
            f"(of {sonnet_self_compare['n_common']} common)"
        )
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
