"""study_a_review_a_context_sensitivity.py

Sensitivity test: does adding PAPER_DELIVERABLE_CONTEXT to REVIEW_A change the
catalog-audit decisions in a substantive way? Runs REVIEW_A twice on the SAME
catalog state (current active_catalog: 42 RG + 138 MG):
  - Run #1: WITHOUT context (allow_paper_context=False) — baseline pre-2026-05-17 behavior
  - Run #2: WITH context (allow_paper_context=True)    — current post-2026-05-17 behavior

Compares decision-distribution shifts (keep/rename/merge/deep_dive per axis)
between the two runs.

Outputs:
  phase2_results/study_a_review_a_no_context.json   — raw Run #1 LLM output
  phase2_results/study_a_review_a_with_context.json — raw Run #2 LLM output
  phase2_results/study_a_context_sensitivity_report.json — head-to-head comparison

Cost: 2 Opus REVIEW_A calls × ~3pp = ~6pp session, ~10 min wall.
NO state mutation: this script does NOT apply decisions, does NOT update
active_catalog, does NOT touch group_remap/path_remap. Pure A/B observation.
"""

from __future__ import annotations
import json
import sys
import uuid
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M

OUT_NO_CTX = M.STEP1.parent / "study_a_review_a_no_context.json"
OUT_W_CTX = M.STEP1.parent / "study_a_review_a_with_context.json"
OUT_REPORT = M.STEP1.parent / "study_a_context_sensitivity_report.json"


def run_one(rg_list, mg_list, rg_counts, mg_counts, themes, allow_paper_context, label):
    sentinel = uuid.uuid4().hex[:12]
    prompt = M.make_review_a_prompt(
        rg_list,
        mg_list,
        rg_counts,
        mg_counts,
        themes,
        sentinel,
        allow_paper_context=allow_paper_context,
    )
    partial_path = M.STEP1 / f"study_a_partial_{label}.txt"
    print(
        f"\n=== STUDY A — Run [{label}] context={allow_paper_context} ===", flush=True
    )
    print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)
    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        f"study_a_{label}",
        partial_path,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(f"  STUDY A run [{label}] FAILED: err={err}", flush=True)
        return None, dur
    try:
        parsed = json.loads(json_part)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}", flush=True)
        recovery_marker = '{"rg_decisions":['
        last = json_part.rfind(recovery_marker)
        if last > 0:
            try:
                parsed = json.loads(json_part[last:])
                print(
                    f"  RECOVERED via rfind-restart (dropped {last} chars)", flush=True
                )
            except Exception as e2:
                print(f"  RECOVERY ALSO FAILED: {e2}", flush=True)
                return None, dur
        else:
            return None, dur
    return parsed, dur


def summarize(parsed):
    """Distribution of decisions per axis."""
    if not parsed:
        return {"error": "no parsed output"}
    out = {"audit_summary": parsed.get("audit_summary", "")[:300]}
    for axis in ("rg", "mg"):
        decisions = parsed.get(f"{axis}_decisions", [])
        c = Counter(d.get("decision", "?") for d in decisions)
        out[f"{axis}_count"] = len(decisions)
        out[f"{axis}_distribution"] = dict(c)
        out[f"{axis}_deep_dive_ids"] = sorted(
            d.get("group_id") for d in decisions if d.get("decision") == "deep_dive"
        )
        out[f"{axis}_merge_pairs"] = sorted(
            (d.get("group_id"), d.get("target_group_id"))
            for d in decisions
            if d.get("decision") == "merge"
        )
        out[f"{axis}_rename_ids"] = sorted(
            d.get("group_id") for d in decisions if d.get("decision") == "rename"
        )
    return out


def main():
    rg_list, mg_list = M._load_active_catalog_or_seed()
    rg_counts, mg_counts, _ = M._compute_group_stats(rg_list, mg_list)
    themes = M._extract_unassigned_themes(
        M._load_unassigned_rows(), k=M.REVIEW_A_THEMES_TOP_K
    )
    print(
        f"loaded: {len(rg_list)} RG + {len(mg_list)} MG, {len(themes)} UNASSIGNED themes"
    )

    parsed_no_ctx, dur_no = run_one(
        rg_list,
        mg_list,
        rg_counts,
        mg_counts,
        themes,
        allow_paper_context=False,
        label="no_context",
    )
    if parsed_no_ctx:
        OUT_NO_CTX.write_text(
            json.dumps(parsed_no_ctx, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  saved {OUT_NO_CTX.name}")

    parsed_w_ctx, dur_w = run_one(
        rg_list,
        mg_list,
        rg_counts,
        mg_counts,
        themes,
        allow_paper_context=True,
        label="with_context",
    )
    if parsed_w_ctx:
        OUT_W_CTX.write_text(
            json.dumps(parsed_w_ctx, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  saved {OUT_W_CTX.name}")

    # Compare
    summary = {
        "n_rg": len(rg_list),
        "n_mg": len(mg_list),
        "dur_no_context_sec": dur_no,
        "dur_with_context_sec": dur_w,
        "no_context": summarize(parsed_no_ctx),
        "with_context": summarize(parsed_w_ctx),
    }
    OUT_REPORT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("STUDY A — distribution comparison")
    print("=" * 70)
    for axis in ("rg", "mg"):
        print(f"\n  {axis.upper()} decision distribution:")
        nc = summary["no_context"].get(f"{axis}_distribution", {})
        wc = summary["with_context"].get(f"{axis}_distribution", {})
        all_keys = sorted(set(nc) | set(wc))
        for k in all_keys:
            print(
                f"    {k:<12} no_context={nc.get(k, 0):>4}   with_context={wc.get(k, 0):>4}"
            )
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
