"""Tier-1A merge audit (post-consolidation_004).

One Opus call: full catalog (37 HC + 39 MC + 6 axes) -> conservative merge candidates.
Report-only: writes phase2_routing_merge_audits/merge_audit_NNN.json. User reviews,
decides what to apply via follow-up consolidation. No auto-apply, no catalog mutation.

Usage:
  python -u phase2_step5b_merge_audit.py
"""

from __future__ import annotations
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step5_opus_routing as R
import phase2_step4_phase2_doublet_llm_grouping as M

MERGE_AUDIT_DIR = R.STEP1 / "phase2_routing_merge_audits"
MERGE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def _next_merge_audit_idx() -> int:
    existing = sorted(MERGE_AUDIT_DIR.glob("merge_audit_*.json"))
    if not existing:
        return 1
    last = existing[-1].stem.rsplit("_", 1)[-1]
    try:
        return int(last) + 1
    except ValueError:
        return len(existing) + 1


def make_merge_audit_prompt(catalog, hc_counts, mc_counts, sentinel: str) -> str:
    catalog_str = R._fmt_catalog_for_routing(catalog, hc_counts, mc_counts)
    return f"""You are conducting a CATALOG-WIDE MERGE AUDIT of the doublet catalog
after consolidation_004. This is the post-routing catalog quality check; the
output drives paper deliverable 2 (large-scale mechanism evaluation).

{M.PAPER_DELIVERABLE_CONTEXT}

============================================================
CURRENT CATALOG (post-consolidation_004)
============================================================

{catalog_str}

============================================================
DEFENDED-HOMOGENEOUS CLASSES (post-consolidation_004)
============================================================

The following classes were defended as single-mechanism-family in consolidation_004
and are paper-deliverable headline findings. They CAN still be merge candidates
(in either direction), but the rationale MUST explicitly address why the
defended-homogeneous claim is wrong if you propose merging them with another class.

  - MC001 (n=175): RLHF — fine-tune on human preferences via learned reward model + PPO
  - MC004 (n=235): Adversarial robustness — generate adversarial inputs/policies + harden
  - MC016 (n=195): Governance/policy mechanisms — legislation, oversight bodies, lifecycle gates

============================================================
YOUR TASK
============================================================

Scan ALL pairs (HC×HC and MC×MC) and identify CONSERVATIVE merge candidates.

A merge candidate is a pair where two classes encode the SAME underlying mechanism
family (for MC) or harm family (for HC), and the distinction between them is an
artifact of historical routing decisions rather than a real conceptual difference.

Conservative bar — ONLY propose a merge when ALL these hold:
  1. The class_descriptions are substantively overlapping (not just topical overlap).
  2. The mechanism family or harm family is genuinely identical — same causal pattern,
     not just same domain.
  3. Merging would NOT create a heterogeneous catch-all (no bundling of ≥2 distinct
     sub-mechanism families into one merged class).
  4. The merged unified description can be written cleanly without "and/or" hedging
     between two genuinely different patterns.

Counter-examples (DO NOT propose):
  - "Both are about adversarial inputs" — but one is training-time hardening and the
    other is inference-time detection. Different mechanism families; keep separate.
  - "Both are governance" — but one is policy/legislation and the other is technical
    audit tooling. Different mechanism families; keep separate.
  - "Both touch RL" — but one is reward shaping and the other is policy regularization.
    Different mechanism families; keep separate.

Genuine merge example: HC025 "Audio/speech perception accuracy ceiling" (n=3) +
HC026 "Audio/speech-perception accuracy ceiling" (n=3) — near-identical naming +
description; clearly the same class accidentally split.

For each merge candidate, output:
  - keep_class_id: the class_id to retain (prefer larger n_members; if equal, lower ID)
  - merge_from_class_id: the class_id whose members move into keep
  - kind: "HC" | "MC"
  - rationale: ≥2 sentences naming what makes them the same mechanism/harm family
  - unified_name: post-merge class name (can be either's existing name or a new one)
  - unified_description: post-merge class description (single coherent paragraph)
  - confidence: "high" | "medium" | "low"

Also identify KEEP_DISTINCT_BUT_NOTE pairs — class pairs that look similar but should
remain separate (e.g. adjacent mechanism families that risk getting conflated by
downstream analysis). For each, briefly state the distinction.

Be CONSERVATIVE — false-positive merges destroy paper deliverable 2's resolution.
False-negatives (failure to merge a near-duplicate) are recoverable in a later pass.
When in doubt, propose KEEP_DISTINCT_BUT_NOTE, not MERGE.

============================================================
OUTPUT FORMAT (STRICT)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.

Schema:

{{
  "merge_candidates": [
    {{"keep_class_id": "HC###",
      "merge_from_class_id": "HC###",
      "kind": "HC",
      "rationale": "...",
      "unified_name": "...",
      "unified_description": "...",
      "confidence": "high"}}
  ],
  "keep_distinct_but_note": [
    {{"class_id_a": "HC###",
      "class_id_b": "HC###",
      "kind": "HC",
      "distinction": "<1-2 sentences naming the genuine mechanism difference>"}}
  ],
  "summary": "<3-5 sentences on catalog merge-cohesion: how many genuine duplicates found, where the catalog is tight, where it has fuzzy boundaries that may need future attention>"
}}END_SENTINEL_{sentinel}

Produce the merge audit now."""


def run_merge_audit():
    catalog = R._load_active_catalog()
    hc_counts, mc_counts = R._compute_class_counts()
    n_hc = len(catalog["harm_classes"])
    n_mc = len(catalog["mechanism_classes"])
    print(
        f"Merge audit input: {n_hc} HC + {n_mc} MC + {len(catalog['axes'])} axes",
        flush=True,
    )

    sentinel = uuid.uuid4().hex[:12]
    prompt = make_merge_audit_prompt(catalog, hc_counts, mc_counts, sentinel)
    audit_idx = _next_merge_audit_idx()
    label = f"merge_audit_{audit_idx:03d}"
    print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)
    print(f"  label: {label}", flush=True)

    partial = R._partial_path(label)
    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        label,
        partial,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(f"MERGE AUDIT FAILED ({err}); partial preserved", flush=True)
        R._mark_partial_failed(partial, reason=f"merge_audit_stream_err={err}")
        return

    try:
        parsed, method = R._robust_json_parse(json_part, '{"merge_candidates":')
        if method != "direct":
            print(f"  RECOVERED via {method}", flush=True)
    except json.JSONDecodeError as e:
        print(f"JSON parse unrecoverable: {e}; partial preserved", flush=True)
        R._mark_partial_failed(partial, reason=f"merge_audit_unrecoverable={e}")
        return

    out = MERGE_AUDIT_DIR / f"merge_audit_{audit_idx:03d}.json"
    M.atomic_write_json(
        out,
        {
            "merge_audit_idx": audit_idx,
            "duration_sec": dur,
            "catalog_state": {
                "n_hc": n_hc,
                "n_mc": n_mc,
                "n_axes": len(catalog["axes"]),
                "post_consolidation_idx": 4,
            },
            "raw_output": parsed,
        },
    )

    merges = parsed.get("merge_candidates", [])
    keep_distinct = parsed.get("keep_distinct_but_note", [])
    print(f"\n=== Merge audit {audit_idx:03d} complete ({dur:.0f}s) ===", flush=True)
    print(f"  Merge candidates: {len(merges)}", flush=True)
    print(f"  Keep-distinct-but-note: {len(keep_distinct)}", flush=True)
    print(f"  Output: {out}", flush=True)
    print(f"\nSummary:\n  {parsed.get('summary', '<none>')}", flush=True)

    if merges:
        print("\nMerge candidates by confidence:", flush=True)
        by_conf = {"high": [], "medium": [], "low": []}
        for m in merges:
            by_conf.setdefault(m.get("confidence", "low"), []).append(m)
        for conf in ("high", "medium", "low"):
            if not by_conf[conf]:
                continue
            print(
                f"\n  [{conf.upper()}] {len(by_conf[conf])} candidate(s):", flush=True
            )
            for m in by_conf[conf]:
                keep = m.get("keep_class_id", "?")
                src = m.get("merge_from_class_id", "?")
                unified = m.get("unified_name", "")[:80]
                print(
                    f"    {src} -> {keep}  ({m.get('kind', '?')}) [{unified}]",
                    flush=True,
                )
                print(f"      {m.get('rationale', '')[:200]}", flush=True)


def make_focused_merge_audit_prompt(
    catalog, class_ids, hc_counts, mc_counts, sentinel: str
) -> str:
    """Focused merge-audit prompt scoped to a subset of class_ids — used to
    de-duplicate sub-classes carved out within a single sweep (chunking
    artifact: chunk N+1 coins a near-duplicate of chunk N's split because it
    didn't know chunk N's name)."""
    # Render only the focused subset
    targets = set(class_ids)
    rows = []
    rows.append("FOCUSED SUBSET UNDER AUDIT:")
    for h in catalog["harm_classes"]:
        if h["class_id"] in targets:
            n = hc_counts.get(h["class_id"], 0)
            rows.append(f"  {h['class_id']} (n={n:>3}): {h['class_name']}")
            rows.append(f"    {M.truncate(h.get('class_description', ''), 240)}")
    for m in catalog["mechanism_classes"]:
        if m["class_id"] in targets:
            n = mc_counts.get(m["class_id"], 0)
            rows.append(f"  {m['class_id']} (n={n:>3}): {m['class_name']}")
            rows.append(f"    {M.truncate(m.get('class_description', ''), 240)}")
    focused_str = "\n".join(rows)

    # Full catalog still shown as context so merges have visibility into
    # adjacent classes (in case some focused sub-class should merge into an
    # existing non-focused class, not just into another focused sibling).
    full_catalog_str = R._fmt_catalog_for_routing(catalog, hc_counts, mc_counts)

    return f"""You are conducting a FOCUSED MERGE AUDIT on a subset of classes
that were just carved out by a single chunked deep-dive sweep. Chunked sweeps
have a known failure mode: chunk N+1 coins a near-duplicate name for a sub-mechanism
that chunk N already proposed (because chunk N+1 did not see chunk N's names at
prompt time). This audit identifies those duplicates and proposes merges so the
catalog is clean before downstream analysis.

{M.PAPER_DELIVERABLE_CONTEXT}

============================================================
FOCUSED SUBSET UNDER AUDIT ({len(targets)} classes)
============================================================

{focused_str}

============================================================
FULL CATALOG (context — for cross-subset merges)
============================================================

{full_catalog_str}

============================================================
YOUR TASK
============================================================

1. Identify CHUNKING-ARTIFACT DUPLICATES within the focused subset — pairs of
   classes that name the SAME mechanism family with slight wording differences
   (e.g. "Calibrated forecasting & superforecaster aggregation for AI" vs
   "Calibrated AI-progress/risk forecasting & prediction-aggregation"). These
   are presumed duplicates UNLESS the class_descriptions reveal a genuinely
   different sub-mechanism.

2. Also identify cross-subset merge candidates — cases where a focused class
   is really the same as a NON-focused existing class.

Less conservative bar than the full catalog audit: when two focused classes
share >=70% of their description content AND the chunking-artifact mode
explains the divergence, propose a MERGE. The base rate of duplicates here is
high (n_chunks × n_sub_families ≈ duplicates).

Conservative bar still applies for cross-subset merges (focused → existing):
only propose if it is genuinely the same mechanism family AND would not create
a heterogeneous catch-all.

For each merge candidate:
  - keep_class_id: retain (prefer lower ID = chunk-1's name, since chunk-1 saw
    no priors and is the "canonical" coining)
  - merge_from_class_id: members move into keep
  - kind: "HC" | "MC"
  - rationale: ≥2 sentences naming the duplicate-pattern AND the mechanism family
  - unified_name: post-merge class name
  - unified_description: post-merge class description (one coherent paragraph)
  - confidence: "high" | "medium" | "low"

Also identify KEEP_DISTINCT pairs in the focused subset that look similar but
are genuinely different sub-mechanisms — briefly state the distinction.

============================================================
OUTPUT FORMAT (STRICT)
============================================================

- ONLY one JSON object. No preamble. No markdown fences.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}`.

Schema:

{{
  "merge_candidates": [
    {{"keep_class_id": "MC###",
      "merge_from_class_id": "MC###",
      "kind": "MC",
      "rationale": "...",
      "unified_name": "...",
      "unified_description": "...",
      "confidence": "high"}}
  ],
  "keep_distinct_but_note": [
    {{"class_id_a": "MC###",
      "class_id_b": "MC###",
      "kind": "MC",
      "distinction": "<1-2 sentences>"}}
  ],
  "summary": "<3-5 sentences on how many duplicates found, expected post-merge count, any cross-subset merges>"
}}END_SENTINEL_{sentinel}

Produce the focused merge audit now."""


def run_focused_merge_audit(class_ids):
    """Focused merge audit scoped to a subset of class_ids. Report-only:
    writes phase2_routing_merge_audits/merge_audit_NNN.json. User reviews,
    decides what to apply."""
    catalog = R._load_active_catalog()
    hc_counts, mc_counts = R._compute_class_counts()
    n_hc = len(catalog["harm_classes"])
    n_mc = len(catalog["mechanism_classes"])
    print(
        f"Focused merge audit input: {len(class_ids)} target classes; "
        f"full catalog has {n_hc} HC + {n_mc} MC + {len(catalog['axes'])} axes",
        flush=True,
    )

    sentinel = uuid.uuid4().hex[:12]
    prompt = make_focused_merge_audit_prompt(
        catalog, class_ids, hc_counts, mc_counts, sentinel
    )
    audit_idx = _next_merge_audit_idx()
    label = f"merge_audit_{audit_idx:03d}_focused"
    print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)
    print(f"  label: {label}", flush=True)
    print(f"  scoped to: {sorted(class_ids)}", flush=True)

    partial = R._partial_path(label)
    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        label,
        partial,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(f"FOCUSED MERGE AUDIT FAILED ({err}); partial preserved", flush=True)
        R._mark_partial_failed(partial, reason=f"focused_merge_audit_stream_err={err}")
        return

    try:
        parsed, method = R._robust_json_parse(json_part, '{"merge_candidates":')
        if method != "direct":
            print(f"  RECOVERED via {method}", flush=True)
    except json.JSONDecodeError as e:
        print(f"JSON parse unrecoverable: {e}; partial preserved", flush=True)
        R._mark_partial_failed(partial, reason=f"focused_merge_audit_unrecoverable={e}")
        return

    out = MERGE_AUDIT_DIR / f"merge_audit_{audit_idx:03d}.json"
    M.atomic_write_json(
        out,
        {
            "merge_audit_idx": audit_idx,
            "audit_kind": "focused",
            "scope_class_ids": sorted(class_ids),
            "duration_sec": dur,
            "catalog_state": {
                "n_hc": n_hc,
                "n_mc": n_mc,
                "n_axes": len(catalog["axes"]),
            },
            "raw_output": parsed,
        },
    )

    merges = parsed.get("merge_candidates", [])
    keep_distinct = parsed.get("keep_distinct_but_note", [])
    print(
        f"\n=== Focused merge audit {audit_idx:03d} complete ({dur:.0f}s) ===",
        flush=True,
    )
    print(f"  Merge candidates: {len(merges)}", flush=True)
    print(f"  Keep-distinct-but-note: {len(keep_distinct)}", flush=True)
    print(f"  Output: {out}", flush=True)
    print(f"\nSummary:\n  {parsed.get('summary', '<none>')}", flush=True)

    if merges:
        print("\nMerge candidates by confidence:", flush=True)
        by_conf = {"high": [], "medium": [], "low": []}
        for m in merges:
            by_conf.setdefault(m.get("confidence", "low"), []).append(m)
        for conf in ("high", "medium", "low"):
            if not by_conf[conf]:
                continue
            print(
                f"\n  [{conf.upper()}] {len(by_conf[conf])} candidate(s):", flush=True
            )
            for m in by_conf[conf]:
                keep = m.get("keep_class_id", "?")
                src = m.get("merge_from_class_id", "?")
                unified = m.get("unified_name", "")[:80]
                print(
                    f"    {src} -> {keep}  ({m.get('kind', '?')}) [{unified}]",
                    flush=True,
                )
                print(f"      {m.get('rationale', '')[:200]}", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class_ids to focus the audit on (e.g. 'MC046,MC047,...'). "
        "If omitted, runs the original catalog-wide merge audit.",
    )
    args = ap.parse_args()
    if args.classes:
        ids = [c.strip() for c in args.classes.split(",") if c.strip()]
        run_focused_merge_audit(ids)
    else:
        run_merge_audit()
