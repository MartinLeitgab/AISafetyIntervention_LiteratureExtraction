"""phase2_step5_opus_routing.py — Opus-only routing pipeline (post-pilot restart).

Pipeline (per restart_plan_2026_05_17.md):
  1. SEED: phase2_pilot_100paths_axis_discovery.json (33 HC + 45 MC + 6 axes)
  2. ROUTING: 75-path batches over 2,672 remaining deduped paths (excluding
     the 100 seed paths). Each batch routes paths to existing classes,
     proposes new (>=3 corpus expected), tags 6 axes per path, emits per-
     assignment confidence, flags catalog-improvement candidates.
  3. CONSOLIDATION: every 10 routing batches, ONE Opus call processes
     accumulated catalog-improvement flags. Applies merges/renames/splits.
  4. FINAL AUDIT: end-of-run single Opus call walks full catalog, fixes
     residual singletons (those still <MIN_GROUP_SIZE at end), proposes
     final axes vocabulary, emits canonical snapshot for paper.

Code-side MIN_GROUP_SIZE enforcement: at batch-end, NEW classes proposed in
this batch with fewer than MIN_GROUP_SIZE assignments in the batch get
DROPPED and their paths force-fit to the closest existing class (by Opus's
own ranking — if it didn't provide a fallback, mark as fallback_needed and
process in next consolidation).

File layout (parallel to old phase2_doublet_*, new namespace phase2_routing_*):
  phase2_routing_active_catalog.json   - current HC + MC + axes state
  phase2_routing_state.json            - batch_counter, last_consolidation_at, etc
  phase2_routing_batches/batch_NNNN.json - per-batch raw outputs
  phase2_routing_flags.jsonl           - accumulated catalog-flag log
  phase2_routing_consolidations/consolidation_NNN.json - per-consolidation outputs
  phase2_routing_final_audit.json      - final audit output
  phase2_routing_assignments.jsonl     - merged per-path assignments (derived)
  phase2_routing_partial.txt           - latest streaming partial

Class A (Opus-only). Per-batch ~14pp; consolidation ~30pp; final ~50pp.
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import random
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import infrastructure from the existing module (streaming, prompt helpers,
# atomic_write_json, fmt_path, SUBTYPE_DEFINITIONS, PAPER_DELIVERABLE_CONTEXT, etc.)
sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M

# ============================================================
# Constants
# ============================================================
ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
# PILOT_FP — discovery JSON used to bootstrap active catalog on first launch.
# After v2 swap (2026-05-18), this MUST point to the v2 pilot (14 HC + 19 MC),
# NOT the deprecated v1 pilot (33 HC + 45 MC). v1 IDs (HC016-HC033, MC020-MC045)
# do NOT exist in the active catalog and would silently corrupt the merged
# assignments jsonl every batch via _rebuild_routing_assignments_jsonl.
PILOT_FP = STEP1 / "phase2_pilot_v2_100paths_discovery.json"
# PILOT_ASSIGNMENTS_FP — flattened jsonl of pilot assignments with full v2 field
# set (fit_score, fit_note, harm_class_status, mechanism_class_status, history).
# Used by _import_pilot_assignments to seed merged jsonl with COMPLETE rows
# instead of dropping the new-architecture fields.
PILOT_ASSIGNMENTS_FP = STEP1 / "phase2_pilot_v2_100paths_assignments.jsonl"
DEDUPED_PATHS = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"

ACTIVE_CATALOG_FP = STEP1 / "phase2_routing_active_catalog.json"
STATE_FP = STEP1 / "phase2_routing_state.json"
BATCH_DIR = STEP1 / "phase2_routing_batches"
FLAGS_LOG_FP = STEP1 / "phase2_routing_flags.jsonl"
CONSOLIDATION_DIR = STEP1 / "phase2_routing_consolidations"
FINAL_AUDIT_FP = STEP1 / "phase2_routing_final_audit.json"
ASSIGNMENTS_FP = STEP1 / "phase2_routing_assignments.jsonl"
PARTIAL_FP = STEP1 / "phase2_routing_partial.txt"
# Path-level reassignments applied AFTER per-batch JSON write — append-only audit
# log read by _rebuild_routing_assignments_jsonl to override the immutable batch
# rows. Lets consolidation/misfit_review mutate path assignments without
# corrupting the per-batch JSON audit trail.
PATH_REASSIGNMENTS_FP = STEP1 / "phase2_routing_path_reassignments.jsonl"
WATCH_ITEMS_FP = STEP1 / "phase2_watch_items.md"
# Sub-channels named by misfit_review/consolidation but not yet at >=3-peer
# quorum. Surfaced into future Opus calls as a watchlist so peers accumulating
# across cycles can be detected.
SURFACED_SUB_CHANNELS_FP = STEP1 / "phase2_routing_surfaced_sub_channels.jsonl"
# Append-only step log — chronological audit trail of every Opus call
# (routing batches, consolidations, sweeps, misfit_reviews, axes_reviews).
# Successor to scattered idx counters in state.json; canonical for "what
# review steps have been completed".
STEP_LOG_FP = STEP1 / "phase2_routing_step_log.jsonl"

BATCH_SIZE = 75
CONSOLIDATION_EVERY = 5
MIN_GROUP_SIZE = 3
OVERSIZE_ALARM = (
    25  # HC/MC above this is a split-eligibility candidate surfaced to every batch
)
ROUTING_RNG_SEED = 20260518


# ============================================================
# State + catalog management
# ============================================================
def _init_active_catalog_from_pilot():
    """Bootstrap active_catalog from the 100-path pilot output. Idempotent."""
    if ACTIVE_CATALOG_FP.exists():
        return
    if not PILOT_FP.exists():
        raise FileNotFoundError(
            f"\nERROR: pilot output required to seed routing catalog.\n"
            f"  Expected: {PILOT_FP}\n"
            f"  Produced by: `python pilot_100_path_axis_discovery.py`\n"
            f"  This script does NOT bootstrap from scratch.\n"
        )
    pilot = json.loads(PILOT_FP.read_text(encoding="utf-8"))
    raw = pilot["raw_output"]
    catalog = {
        "source": "pilot_100paths",
        "harm_classes": raw["harm_classes"],
        "mechanism_classes": raw["mechanism_classes"],
        "axes": raw["axes"],
        "init_assignments_count": len(raw.get("assignments", [])),
    }
    M.atomic_write_json(ACTIVE_CATALOG_FP, catalog)
    print(
        f"  bootstrapped active catalog from pilot: "
        f"{len(catalog['harm_classes'])} HC + "
        f"{len(catalog['mechanism_classes'])} MC + "
        f"{len(catalog['axes'])} axes",
        flush=True,
    )


def _load_active_catalog():
    if not ACTIVE_CATALOG_FP.exists():
        _init_active_catalog_from_pilot()
    return json.loads(ACTIVE_CATALOG_FP.read_text(encoding="utf-8"))


def _save_active_catalog(catalog, kind_suffix=""):
    """Save catalog with timestamped backup before overwrite."""
    import shutil

    if ACTIVE_CATALOG_FP.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = ACTIVE_CATALOG_FP.with_suffix(f".pre_{kind_suffix}_{ts}.json")
        try:
            shutil.copy2(ACTIVE_CATALOG_FP, backup)
        except Exception as e:
            print(f"  WARN: catalog backup failed: {e}", flush=True)
    M.atomic_write_json(ACTIVE_CATALOG_FP, catalog)


def _load_state():
    if STATE_FP.exists():
        return json.loads(STATE_FP.read_text(encoding="utf-8"))
    return {
        "total_batches_run": 0,
        "last_consolidation_at_batch_idx": -1,
        "total_consolidations_run": 0,
        "pilot_assignments_seeded": False,
    }


def _save_state(state):
    M.atomic_write_json(STATE_FP, state)


def _import_pilot_assignments():
    """Read the v2 pilot's flattened assignments jsonl as-is — preserves the full
    v2 field set (fit_score, fit_note, harm_class_status, mechanism_class_status,
    history, axes). Called by _rebuild_routing_assignments_jsonl on every batch;
    must read v2 (14 HC + 19 MC) NOT v1 to avoid silent jsonl corruption.

    FAIL-FAST: if v2 flattened pilot is missing, crash with clear error rather
    than silent fall-back to v1 (would re-introduce HC016-HC033 orphan IDs)."""
    if not PILOT_ASSIGNMENTS_FP.exists():
        raise FileNotFoundError(
            f"\nERROR: v2 pilot assignments missing at {PILOT_ASSIGNMENTS_FP}\n"
            f"  This is produced by `python flatten_pilot_v2_to_jsonl.py` and is\n"
            f"  the canonical seed for the merged routing assignments jsonl.\n"
            f"  This script does NOT fall back to v1 (phase2_pilot_100paths_\n"
            f"  axis_discovery.json) — falling back would silently re-introduce\n"
            f"  v1 orphan IDs (HC016-HC033, MC020-MC045) that don't exist in\n"
            f"  the active v2 catalog (14 HC + 19 MC).\n"
        )
    rows = []
    for line in PILOT_ASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ============================================================
# Class ID allocation + path reassignment log
# ============================================================
def _next_class_id(catalog, kind):
    """Allocate next HC### / MC### id given current catalog. kind in {"HC","MC"}."""
    key = "harm_classes" if kind == "HC" else "mechanism_classes"
    existing = [
        int(c["class_id"][2:])
        for c in catalog.get(key, [])
        if c["class_id"][2:].isdigit()
    ]
    return f"{kind}{(max(existing, default=0) + 1):03d}"


_SENTINEL_NOOP = "__NOOP__"  # internal sentinel meaning "don't touch this side"
_SENTINEL_UNASSIGN = "__UNASSIGN__"  # set side to None (mark unassigned)


def _append_path_reassignment(
    path_id,
    source,
    new_hc=_SENTINEL_NOOP,
    new_mc=_SENTINEL_NOOP,
    rationale="",
    batch_idx=None,
):
    """Append a path reassignment record to the override log. Read by
    _rebuild_routing_assignments_jsonl to apply AFTER per-batch sources are loaded.

    Per-side semantics:
      new_hc / new_mc = _SENTINEL_NOOP (default): do not modify that side
      new_hc / new_mc = _SENTINEL_UNASSIGN: set side to None (unassigned)
      new_hc / new_mc = "HC###" / "MC###": set to that class id
    """
    record = {
        "path_id": path_id,
        "source": source,
        "rationale": rationale[:300],
    }
    if new_hc != _SENTINEL_NOOP:
        record["harm_class_id"] = None if new_hc == _SENTINEL_UNASSIGN else new_hc
    if new_mc != _SENTINEL_NOOP:
        record["mechanism_class_id"] = None if new_mc == _SENTINEL_UNASSIGN else new_mc
    if batch_idx is not None:
        record["applied_at_batch_idx"] = batch_idx
    with open(PATH_REASSIGNMENTS_FP, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_path_reassignments():
    """Read full reassignment log. Returns {path_id -> latest record dict}.
    Latest entry wins (append-only log; later entries override earlier)."""
    if not PATH_REASSIGNMENTS_FP.exists():
        return {}
    latest = {}
    for line in PATH_REASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            latest[r["path_id"]] = r
    return latest


# ============================================================
# Robust JSON parse for Opus outputs
# ============================================================
def _robust_json_parse(payload, expected_start_pattern=None):
    """Parse Opus output JSON with cascading recovery strategies.

    Common Opus output corruption modes (observed 2026-05-18):
      (a) Trailing brace dropped: ends `"}]}` instead of `"}]}}`
      (b) Multi-bracket trailing missing
      (c) Mid-stream restart pollution (Opus restarted the JSON; need rfind)

    Strategy order:
      1. Direct parse
      2. Append `}` (single missing close)
      3. Append `]}` (missing array+object close)
      4. Append `"}]}` (missing string close + array+object close)
      5. If `expected_start_pattern` provided: rfind from latest occurrence
         and retry with same trailing-fix cascade

    Returns: (parsed_dict, recovery_method_str) on success, raises last
    JSONDecodeError if all strategies fail.
    """
    import json as _json

    # Strategy 1: direct
    try:
        return _json.loads(payload), "direct"
    except _json.JSONDecodeError as e0:
        last_err = e0
    # Strategy 2/3/4: trailing-bracket appends
    for suffix, label in [
        ("}", "trailing-brace"),
        ("]}", "trailing-array-brace"),
        ('"}]}', "trailing-string-array-brace"),
    ]:
        try:
            return _json.loads(payload + suffix), f"appended-{label}"
        except _json.JSONDecodeError as e:
            last_err = e
    # Strategy 5/6/7: unterminated-string at the tail (Opus emitted a string
    # value without a closing `"` before the final `}`). Insert `"` BEFORE the
    # trailing close brackets.
    for tail, fix, label in [
        ("}", '"}', "insert-quote-before-trailing-brace"),
        ("]}", '"]}', "insert-quote-before-trailing-array-brace"),
        ("}]}", '"}]}', "insert-quote-before-trailing-obj-array-brace"),
    ]:
        if payload.endswith(tail):
            candidate = payload[: -len(tail)] + fix
            try:
                return _json.loads(candidate), f"recovered-{label}"
            except _json.JSONDecodeError as e:
                last_err = e
    # Strategy 5: rfind-restart with trailing-fix cascade
    if expected_start_pattern:
        last = payload.rfind(expected_start_pattern)
        if last > 0:
            restart_payload = payload[last:]
            for suffix, label in [
                ("", "direct"),
                ("}", "trailing-brace"),
                ("]}", "trailing-array-brace"),
                ('"}]}', "trailing-string-array-brace"),
            ]:
                try:
                    return (
                        _json.loads(restart_payload + suffix),
                        f"rfind-restart-{label} (dropped {last} chars)",
                    )
                except _json.JSONDecodeError as e:
                    last_err = e
    raise last_err


# ============================================================
# Path universe
# ============================================================
def _load_deduped_paths():
    paths = []
    with open(DEDUPED_PATHS, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                d = json.loads(line)
                d["path_id"] = f"path_{i:05d}_dedup"
                paths.append(d)
    return paths


def _get_done_path_ids():
    """All path_ids already routed (pilot + saved routing batches)."""
    done = set()
    pilot = json.loads(PILOT_FP.read_text(encoding="utf-8"))
    for a in pilot["raw_output"].get("assignments", []):
        if a.get("path_id"):
            done.add(a["path_id"])
    if BATCH_DIR.exists():
        for bf in sorted(BATCH_DIR.glob("batch_*.json")):
            d = json.loads(bf.read_text(encoding="utf-8"))
            for a in d.get("resolved_assignments", []):
                if a.get("path_id"):
                    done.add(a["path_id"])
    return done


# ============================================================
# Routing prompt
# ============================================================
def _fmt_catalog_for_routing(catalog, hc_counts, mc_counts):
    """Compact catalog rendering for routing-batch prompt."""
    rows = []
    rows.append(
        "AXES (controlled vocabularies, use 'OTHER:<free_text>' if no value fits):"
    )
    for ax in catalog.get("axes", []):
        rows.append(
            f"  - {ax['axis_name']} ({ax['axis_kind']}): {ax.get('values', [])}"
        )
    rows.append(f"\nHARM CLASSES (n={len(catalog['harm_classes'])}):")
    for h in catalog["harm_classes"]:
        n = hc_counts.get(h["class_id"], 0)
        flag = " [CAP-GAP]" if h.get("is_capability_gap") else ""
        rows.append(f"  {h['class_id']}{flag} (n={n:>3}): {h['class_name'][:75]}")
        rows.append(f"    {M.truncate(h.get('class_description', ''), 200)}")
    rows.append(f"\nMECHANISM CLASSES (n={len(catalog['mechanism_classes'])}):")
    for m in catalog["mechanism_classes"]:
        n = mc_counts.get(m["class_id"], 0)
        rows.append(f"  {m['class_id']} (n={n:>3}): {m['class_name'][:75]}")
        rows.append(f"    {M.truncate(m.get('class_description', ''), 200)}")
    return "\n".join(rows)


def _load_watch_items():
    """Read the canonical watch-items markdown file. Returns string or empty."""
    if WATCH_ITEMS_FP.exists():
        return WATCH_ITEMS_FP.read_text(encoding="utf-8")
    return ""


def _append_to_watch_items(pass_kind, pass_idx, summary_text, extra_lines=None):
    """Append a pass-summary block to watch_items so future routing batches
    see what prior passes observed (HC/MC catch-alls, recurring drift patterns,
    notable reassign rationales). Idempotent guard: skips if the exact pass-id
    header already exists in the file.
    """
    import datetime

    if not summary_text and not extra_lines:
        return
    header = f"## auto-appended {datetime.date.today().isoformat()} — {pass_kind} #{pass_idx:03d}"
    body = WATCH_ITEMS_FP.read_text(encoding="utf-8") if WATCH_ITEMS_FP.exists() else ""
    if header in body:
        return  # already appended (idempotent)
    block = [header, ""]
    if summary_text:
        block.append(summary_text.strip())
    if extra_lines:
        block.extend(extra_lines)
    block.append("")
    insertion = "\n".join(block)
    # Insert BEFORE the "## Format for new entries" footer if it exists, else append
    marker = "## Format for new entries"
    if marker in body:
        body = body.replace(marker, insertion + "\n" + marker)
    else:
        body = body.rstrip() + "\n\n" + insertion + "\n"
    WATCH_ITEMS_FP.write_text(body, encoding="utf-8")


_OOS_KEYWORDS = (
    "non-ai-safety",
    "non-ai safety",
    "non ai safety",
    "outside ai-safety scope",
    "outside ai safety scope",
    "outside ai-safety",
    "stay unassigned",
    "permanent unassigned",
    "no ai-safety harm chain",
    "no ai safety harm chain",
    "not ai-safety",
    "not ai safety",
    "non-safety domain",
    "non-safety domains",
    "non-ai cybersec",
    "non-ai cyber",
    "outside the ai safety scope",
    "outside ai-safety analysis",
)


def _detect_oos_from_text(text):
    """Heuristic: does the rationale text declare this sub-channel as
    out-of-AI-safety-scope (= 'do not propose as new class even at quorum')?

    Returns True if any OOS keyword is present. False otherwise. Conservative:
    only fires on explicit OOS language, not on borderline cases.
    """
    t = (text or "").lower()
    return any(k in t for k in _OOS_KEYWORDS)


def _append_surfaced_sub_channels(parsed, source_label, batch_idx_or_step=None):
    """Persist any 'surfaced_sub_channels' Opus emitted (named gaps that did
    not yet reach >=3-peer quorum) to a sidecar file. Future Opus calls load
    this as a WATCHLIST so peers accumulating across cycles get detected.

    Each entry can carry a `confirmed_out_of_scope: bool` flag (set when
    Opus explicitly emits it, OR auto-detected from OOS keywords in the
    rationale). OOS entries are rendered separately in the prompt loader
    as 'DO NOT PROPOSE — already adjudicated as out-of-AI-safety-scope',
    preventing them from re-surfacing as actionable new-class candidates
    in future review cycles.

    Idempotent on (source_label, sub_channel name). Existing entries from the
    same source are deduped by name."""
    import datetime

    subs = parsed.get("surfaced_sub_channels", []) or []
    if not subs:
        return 0
    # De-dupe within this source by name (lowercased)
    seen_in_call = set()
    written = 0
    iso_now = datetime.datetime.now().isoformat(timespec="seconds")
    with open(SURFACED_SUB_CHANNELS_FP, "a", encoding="utf-8") as f:
        for s in subs:
            name = (s.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in seen_in_call:
                continue
            seen_in_call.add(key)
            rationale_text = s.get("rationale", "")[:400]
            oos_explicit = bool(s.get("confirmed_out_of_scope", False))
            oos_detected = _detect_oos_from_text(rationale_text)
            entry = {
                "first_seen_at": iso_now,
                "source": source_label,
                "name": name,
                "side": s.get("side", ""),
                "rationale": rationale_text,
                "named_peer_path_ids": s.get("named_peer_path_ids", []) or [],
                "n_peers_in_corpus_so_far": int(s.get("n_peers_in_corpus_so_far") or 0),
                "min_required": int(s.get("min_required") or MIN_GROUP_SIZE),
                "confirmed_out_of_scope": oos_explicit or oos_detected,
                "oos_source": (
                    "opus_explicit"
                    if oos_explicit
                    else ("auto_detected_keywords" if oos_detected else None)
                ),
            }
            if batch_idx_or_step is not None:
                entry["batch_idx_or_step"] = batch_idx_or_step
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
    return written


def _load_surfaced_sub_channels_for_prompt(only_last_source=None):
    """Return a formatted string of the surfaced sub-channels watchlist for
    inclusion in future Opus prompts. Coalesces by name (case-insensitive),
    tracks cumulative named peers + last source.

    If `only_last_source` is provided, returns ONLY entries from that source
    label (used to inject 'results from the last misfit review' specifically
    into the next call, per user requirement)."""
    if not SURFACED_SUB_CHANNELS_FP.exists():
        return "(no surfaced sub-channels recorded yet)"
    by_name = {}
    for line in SURFACED_SUB_CHANNELS_FP.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if only_last_source and e.get("source") != only_last_source:
            continue
        key = (e.get("name") or "").strip().lower()
        if not key:
            continue
        existing = by_name.get(key)
        peers = set((existing or {}).get("named_peer_path_ids", [])) | set(
            e.get("named_peer_path_ids", []) or []
        )
        # Sticky OOS: once any observation of this name is OOS, the coalesced
        # entry stays OOS permanently (one downstream review having adjudicated
        # it as out-of-scope is sufficient — re-surfacing in a later review
        # does NOT clear the OOS flag).
        oos_now = bool(e.get("confirmed_out_of_scope", False))
        oos_prior = bool((existing or {}).get("confirmed_out_of_scope", False))
        by_name[key] = {
            "name": e.get("name"),
            "side": e.get("side"),
            "rationale": e.get("rationale", "")[:300],
            "named_peer_path_ids": sorted(peers),
            "n_peers_in_corpus_so_far": max(
                (existing or {}).get("n_peers_in_corpus_so_far", 0),
                e.get("n_peers_in_corpus_so_far", 0),
            ),
            "min_required": e.get("min_required", MIN_GROUP_SIZE),
            "first_seen_at": (existing or {}).get("first_seen_at")
            or e.get("first_seen_at"),
            "last_observed_source": e.get("source"),
            "confirmed_out_of_scope": oos_now or oos_prior,
        }
    if not by_name:
        return "(no surfaced sub-channels recorded yet)"

    # Split into ACTIVE (in-scope, accumulate peers toward quorum) vs CONFIRMED
    # OUT-OF-SCOPE (already adjudicated as non-AI-safety; DO NOT propose).
    active = []
    oos = []
    for v in sorted(
        by_name.values(), key=lambda x: (-x["n_peers_in_corpus_so_far"], x["name"])
    ):
        if v.get("confirmed_out_of_scope"):
            oos.append(v)
        else:
            active.append(v)

    out_lines = []
    if active:
        out_lines.append(
            "ACTIVE WATCHLIST (in-scope sub-channels — accumulate peers across "
            "cycles; emit PROPOSE_NEW / SPLIT_OUT once cumulative peers ≥ "
            "min_required):"
        )
        for v in active:
            gap = max(0, v["min_required"] - len(v["named_peer_path_ids"]))
            peers_repr = (
                ", ".join(v["named_peer_path_ids"])
                if v["named_peer_path_ids"]
                else "(none named)"
            )
            out_lines.append(
                f"- [{v['side']}] {v['name']} — peers so far "
                f"({len(v['named_peer_path_ids'])}/{v['min_required']}, "
                f"need {gap} more): {peers_repr}"
                + (f"\n    rationale: {v['rationale']}" if v.get("rationale") else "")
                + f"\n    last seen in: {v['last_observed_source']}"
            )
    if oos:
        if out_lines:
            out_lines.append("")
        out_lines.append(
            "CONFIRMED OUT-OF-SCOPE — DO NOT PROPOSE as a new HC or MC class, "
            "EVEN IF cumulative peers ≥ min_required. These sub-channels have "
            "been adjudicated in prior reviews as falling outside the "
            "AI-safety paper scope (paths matching them stay UNASSIGNED):"
        )
        for v in oos:
            peer_count = len(v["named_peer_path_ids"])
            out_lines.append(
                f"- [{v['side']}] {v['name']} — {peer_count} peers observed; "
                f"OUT-OF-SCOPE"
                + (f"\n    rationale: {v['rationale']}" if v.get("rationale") else "")
            )
    return (
        "\n".join(out_lines) if out_lines else "(no surfaced sub-channels recorded yet)"
    )


def run_backfill_oos():
    """One-shot maintenance: re-tag every entry in
    phase2_routing_surfaced_sub_channels.jsonl with confirmed_out_of_scope
    based on keyword detection over its rationale text. Existing OOS flags
    are preserved (OR semantics — never downgrade). Writes a .bak alongside
    the file before rewriting in place.
    """
    import datetime
    import shutil

    if not SURFACED_SUB_CHANNELS_FP.exists():
        print(
            f"no watchlist at {SURFACED_SUB_CHANNELS_FP}; nothing to backfill",
            flush=True,
        )
        return
    bak = SURFACED_SUB_CHANNELS_FP.with_suffix(
        SURFACED_SUB_CHANNELS_FP.suffix
        + ".pre_oos_backfill_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".bak"
    )
    shutil.copy2(SURFACED_SUB_CHANNELS_FP, bak)
    print(f"backed up watchlist to {bak.name}", flush=True)

    entries = []
    for line in SURFACED_SUB_CHANNELS_FP.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        entries.append(e)

    n_before_oos = sum(1 for e in entries if e.get("confirmed_out_of_scope"))
    new_tags = []
    for e in entries:
        prior = bool(e.get("confirmed_out_of_scope", False))
        detected = _detect_oos_from_text(e.get("rationale", ""))
        final_flag = prior or detected
        e["confirmed_out_of_scope"] = final_flag
        if not prior and detected:
            e["oos_source"] = "auto_detected_keywords_backfill"
            new_tags.append(e.get("name"))
        elif "oos_source" not in e:
            e["oos_source"] = "opus_explicit" if prior else None

    with open(SURFACED_SUB_CHANNELS_FP, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n_after_oos = sum(1 for e in entries if e.get("confirmed_out_of_scope"))
    print(f"backfill complete: {n_before_oos} -> {n_after_oos} OOS entries", flush=True)
    print(f"newly tagged ({len(new_tags)}):", flush=True)
    for name in sorted(set(new_tags)):
        print(f"  - {name}", flush=True)


def _append_step_log(step_type, summary, extra=None):
    """Append a chronological entry to the step log — every Opus call gets
    one line. Successor to scattered idx counters in state.json."""
    import datetime

    entry = {
        "step_idx": _next_step_idx(),
        "step_type": step_type,
        "logged_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
    }
    if extra:
        entry.update(extra)
    with open(STEP_LOG_FP, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry["step_idx"]


def _next_step_idx():
    if not STEP_LOG_FP.exists():
        return 1
    max_idx = 0
    for line in STEP_LOG_FP.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            if e.get("step_idx", 0) > max_idx:
                max_idx = e["step_idx"]
        except json.JSONDecodeError:
            continue
    return max_idx + 1


def _build_attention_queue(catalog, hc_counts, mc_counts, max_rows_per_section=20):
    """Surface to Opus the paths that prior batches/pilot left as 'open issues'.
    The queue is rebuilt each batch from the merged assignments jsonl so the
    routing prompt always shows the current state of:
      - paths with harm_class_id=None (unassigned at harm side, awaiting peers)
      - paths with mechanism_class_id=None (unassigned at mechanism side)
      - paths with fit_score <= 3 (low-fit assignments — reassign candidates)
      - paths flagged reassign_pending (from manual audit)
      - HC/MC classes with count < MIN_GROUP_SIZE (below-min, force-merge candidates)

    Returns a markdown block. Returns "(empty)" if no items.
    """
    if not ASSIGNMENTS_FP.exists():
        return "(no prior assignments yet)"
    rows = []
    for line in ASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        return "(no prior assignments yet)"

    out = []
    # 0) Oversized classes — split-eligibility alarm. Surfaced FIRST so Opus
    # treats in-batch split proposals as a default action, not a deferred one.
    oversized_hc = sorted(
        [(hid, n) for hid, n in hc_counts.items() if n >= OVERSIZE_ALARM],
        key=lambda kv: -kv[1],
    )
    oversized_mc = sorted(
        [(mid, n) for mid, n in mc_counts.items() if n >= OVERSIZE_ALARM],
        key=lambda kv: -kv[1],
    )
    if oversized_hc or oversized_mc:
        out.append(
            f"OVERSIZED CLASSES — split-eligibility ALARM (threshold {OVERSIZE_ALARM}):\n"
            f"These classes are large enough that latent sub-channels likely exist.\n"
            f"If you see >=3 paths in THIS batch fitting a coherent sub-channel of any\n"
            f"oversized class, PROPOSE_NEW_HC / PROPOSE_NEW_MC in-batch NOW (do not\n"
            f"defer to consolidation). The new class needs ONLY 3 members from this\n"
            f"batch to survive MIN_GROUP_SIZE; existing members can be moved later via\n"
            f"reassign_candidates flag. Splitting catch-alls early avoids massive\n"
            f"re-routing cost at consolidation."
        )
        for hid, n in oversized_hc:
            name = next(
                (
                    h["class_name"]
                    for h in catalog.get("harm_classes", [])
                    if h["class_id"] == hid
                ),
                "?",
            )
            out.append(f"  - {hid} ({n} members): {name}")
        for mid, n in oversized_mc:
            name = next(
                (
                    m["class_name"]
                    for m in catalog.get("mechanism_classes", [])
                    if m["class_id"] == mid
                ),
                "?",
            )
            out.append(f"  - {mid} ({n} members): {name}")
        out.append("")

    # 1) Unassigned (HC and/or MC) — these are paths awaiting peers to form a group
    unassigned_hc = [
        r
        for r in rows
        if r.get("harm_class_status") == "unassigned"
        or (
            r.get("harm_class_id") is None
            and r.get("harm_class_status") != "force_fit_pending"
        )
    ]
    unassigned_mc = [
        r
        for r in rows
        if r.get("mechanism_class_status") == "unassigned"
        or (
            r.get("mechanism_class_id") is None
            and r.get("mechanism_class_status") != "force_fit_pending"
        )
    ]
    if unassigned_hc:
        out.append(
            f"PATHS WITH harm_class UNASSIGNED (n={len(unassigned_hc)}) — "
            f"if any current batch path shares a failure mode with these, "
            f"consider proposing a new HC to group them together:"
        )
        for r in unassigned_hc[:max_rows_per_section]:
            out.append(f"  - {r['path_id']}: {(r.get('fit_note') or '')[:120]}")
        if len(unassigned_hc) > max_rows_per_section:
            out.append(f"  ... and {len(unassigned_hc) - max_rows_per_section} more")
        out.append("")

    if unassigned_mc:
        out.append(
            f"PATHS WITH mechanism_class UNASSIGNED (n={len(unassigned_mc)}) — "
            f"if current batch paths share the intervention pattern, propose "
            f"a new MC to group them:"
        )
        for r in unassigned_mc[:max_rows_per_section]:
            out.append(f"  - {r['path_id']}: {(r.get('fit_note') or '')[:120]}")
        if len(unassigned_mc) > max_rows_per_section:
            out.append(f"  ... and {len(unassigned_mc) - max_rows_per_section} more")
        out.append("")

    # 2) Low fit_score paths — reassign candidates
    low_fit = [
        r for r in rows if r.get("fit_score") is not None and r["fit_score"] <= 3
    ]
    if low_fit:
        out.append(
            f"PATHS WITH LOW fit_score (<=3, n={len(low_fit)}) — these are "
            f"forced or borderline fits. If a new HC/MC emerging in current "
            f"batch fits one of these better, flag reassign_candidates:"
        )
        for r in low_fit[:max_rows_per_section]:
            out.append(
                f"  - {r['path_id']} (HC={r.get('harm_class_id')} "
                f"MC={r.get('mechanism_class_id')} fit={r.get('fit_score')}): "
                f"{(r.get('fit_note') or '')[:120]}"
            )
        if len(low_fit) > max_rows_per_section:
            out.append(f"  ... and {len(low_fit) - max_rows_per_section} more")
        out.append("")

    # 3) Reassign-pending flagged paths (from manual audit)
    reassign = [r for r in rows if r.get("reassign_pending")]
    if reassign:
        out.append(
            f"PATHS FLAGGED reassign_pending (n={len(reassign)}) — manual "
            f"audit flagged these as misfits; re-evaluate when seen:"
        )
        for r in reassign:
            out.append(
                f"  - {r['path_id']} (HC={r.get('harm_class_id')} "
                f"MC={r.get('mechanism_class_id')}): "
                f"{(r.get('history', [{}])[-1].get('edit', '') if r.get('history') else '')[:120]}"
            )
        out.append("")

    # 4) Below-min classes
    below_min_hc = [hid for hid, n in hc_counts.items() if n < MIN_GROUP_SIZE]
    below_min_mc = [mid for mid, n in mc_counts.items() if n < MIN_GROUP_SIZE]
    if below_min_hc:
        out.append(
            f"HARM CLASSES BELOW MIN_GROUP_SIZE ({MIN_GROUP_SIZE}): "
            f"{[(h, hc_counts[h]) for h in below_min_hc]} — "
            f"watch for peer paths in this batch; otherwise candidates for force_merge."
        )
    if below_min_mc:
        out.append(
            f"MECHANISM CLASSES BELOW MIN_GROUP_SIZE ({MIN_GROUP_SIZE}): "
            f"{[(m, mc_counts[m]) for m in below_min_mc]} — same."
        )

    # 5) Heuristic misfit candidates (regenerated each batch; surfaces paths
    #    that Opus did NOT self-flag but where keyword-overlap suggests poor fit)
    heuristic_path = STEP1 / "phase2_routing_heuristic_misfits.json"
    if heuristic_path.exists():
        try:
            hm = json.loads(heuristic_path.read_text(encoding="utf-8"))
            top_hc = hm.get("top_hc_misfits", [])[:max_rows_per_section]
            top_mc = hm.get("top_mc_misfits", [])[:max_rows_per_section]
            if top_hc:
                out.append("")
                out.append(
                    "HEURISTIC HARM-CLASS MISFIT CANDIDATES (Class B keyword-overlap audit; "
                    "Opus did NOT self-flag these but their risk text overlaps poorly with the "
                    "assigned HC's definition — re-evaluate when seen):"
                )
                for r in top_hc:
                    out.append(
                        f"  - {r['path_id']} (HC={r['class_id']} overlap={r['risk_overlap']}): "
                        f"{r.get('risk_text_head', '')[:120]}"
                    )
            if top_mc:
                out.append("")
                out.append("HEURISTIC MECH-CLASS MISFIT CANDIDATES (same):")
                for r in top_mc:
                    out.append(
                        f"  - {r['path_id']} (MC={r['class_id']} overlap={r['mech_overlap']}): "
                        f"{r.get('mech_text_head', '')[:120]}"
                    )
        except Exception as e:
            out.append(f"(heuristic misfit file read error: {e})")

    return "\n".join(out) if out else "(no open attention items)"


def make_routing_prompt(paths, catalog, hc_counts, mc_counts, node_attrs, sentinel):
    body = "\n\n".join(M.fmt_path(p, node_attrs) for p in paths)
    n = len(paths)
    catalog_str = _fmt_catalog_for_routing(catalog, hc_counts, mc_counts)
    watch_str = _load_watch_items() or "(no watch items recorded yet)"
    attn_str = _build_attention_queue(catalog, hc_counts, mc_counts)
    sub_channels_str = _load_surfaced_sub_channels_for_prompt()
    return f"""You are routing AI-safety research paths into a faceted-classification catalog.

{M.PAPER_DELIVERABLE_CONTEXT}

============================================================
PATH ROLES CONTEXT
============================================================

{M.SUBTYPE_DEFINITIONS}

============================================================
ACTIVE WATCH-ITEMS (running audit log — consult before assigning)
============================================================

{watch_str}

============================================================
SURFACED SUB-CHANNELS WATCHLIST (catalog gaps awaiting quorum)

When routing a path, EXPLICITLY check if it matches any of these named gaps.
If yes:
  (a) Route the path to the closest existing class as normal (so it has a home)
  (b) Add its path_id to flags.reassign_candidates with a note "matches
      watchlist sub-channel <name>" so the next consolidation can accumulate
      the peer and split when the cumulative count hits >=3.
  (c) If THIS batch alone produces >=3 named peers matching a watchlist
      sub-channel, propose a NEW class via the new_HCs/new_MCs proposal slot
      with all in-batch peers cited (the watchlist's prior peers will be
      added by the resolver).
============================================================

{sub_channels_str}

============================================================
ATTENTION QUEUE (open issues from prior batches — re-use opportunities)
============================================================

{attn_str}

If any path in the CURRENT batch shares a failure mode or mechanism pattern with
the unassigned / low-fit paths above, that is a SIGNAL to propose a new HC/MC
that groups them together (the prior path becomes the >=2nd member, your current
becomes >=3rd, the new group survives MIN_GROUP_SIZE). Also: if a new class
emerges this batch that fits a low-fit prior path better than its current
assignment, flag reassign_candidates so consolidation can move it.

NO-HETEROGENEOUS-CATCH-ALL POLICY (HARD RULE — applies to every HC and MC):

Every HC must name a single coherent harm-mechanism family. Every MC must name
a single coherent intervention-mechanism family. The trigger for splitting is
HETEROGENEITY, not size — a class of n=50 or n=100 is FULLY ACCEPTABLE if all
its members share one mechanism family (e.g., one coherent intervention pattern
that genuinely recurs across many papers). LARGE COHERENT CLASSES ARE GOOD —
they ARE the paper's headline finding (mechanism X is densely populated).

What IS forbidden: classes that bundle ≥2 distinct sub-mechanism families with
≥3 members each. Such classes contaminate the deliverable-2 catalog (the
distribution becomes meaningless) and hide deliverable-3 novel-intervention
candidates (a thin sub-mechanism is invisible when buried under a fat bucket
of unrelated mechanisms).

`unassigned` is the ONLY legitimate residual bucket for paths that don't fit
any coherent class. NEVER route a poorly-fitting path into an HC or MC just
because the class name is "general enough" to absorb it.

IN-BATCH AUDIT POLICY (when OVERSIZED CLASSES alarm fires):

The alarm is an AUDIT trigger, not a split mandate. For each alarmed class,
evaluate:

  Decision 1 — DEFEND AS HOMOGENEOUS
    Take this when: members all share one mechanism family. Argue concretely:
    cite 2-3 representative member paths and identify the shared mechanism
    family in 1-2 sentences. Put this in catalog_flags.summary under a
    `defended_classes` key. The class stays as-is. THIS IS A LEGITIMATE
    OUTCOME — large coherent classes ARE the paper's finding.

  Decision 2 — SPLIT (in-batch)
    Take this when: class spans ≥2 sub-mechanism families AND ≥3 paths in
    THIS batch fit one sub-channel cleanly. Steps:
    (a) Identify ≥3 paths in THIS batch sharing a finer mechanism pattern.
    (b) PROPOSE_NEW_HC or PROPOSE_NEW_MC for that sub-channel via the `new`
        slot, naming the in-batch member paths.
    (c) In catalog_flags.reassign_candidates, list 2-5 prior members of the
        oversized class that are candidates to migrate to the new sub-class;
        consolidation will adjudicate.

  Decision 3 — PARK SUB-CHANNEL
    Take this when: a sub-channel exists but has <3 in-batch members.
    Route those <3 candidates to `unassigned` with fit_note tagging the
    sub-channel name (e.g., "subchannel=value-mis-specification of HC002")
    so consolidation can promote when corpus scale provides peers.

FORBIDDEN outcome: defending a class as "broad-but-coherent" / "general enough
to cover diverse paths" / "structural / sociotechnical / field-building bucket".
That phrasing IS the catch-all anti-pattern. The test is one mechanism family
or not — there is no "broad bucket" middle ground.

Splitting heterogeneous classes earlier prevents bloat AND avoids massive re-
routing cost when consolidation eventually splits at n=100+. Catalog mutation
IS the default action when heterogeneity evidence is in hand.

============================================================
CURRENT CATALOG
============================================================

{catalog_str}

============================================================
YOUR TASK — assign all {n} paths + flag catalog improvements
============================================================

For EACH input path, output:
  - path_id (verbatim)
  - harm_class:      {{"existing": "HC###"}} OR {{"new": {{"class_name": "...", "class_description": "...", "is_capability_gap": bool}}}} OR {{"unassigned": true, "reason": "<1 clause>"}}
  - mechanism_class: {{"existing": "MC###"}} OR {{"new": {{"class_name": "...", "class_description": "..."}}}} OR {{"unassigned": true, "reason": "<1 clause>"}}
  - axis_values: {{<axis_name>: <value from catalog vocab OR "OTHER:<free_text>">, ...}}
                 — one value per axis defined in CURRENT CATALOG
  - confidence: 1 (low) to 5 (high) — how clearly you READ the path's causal chain
                (epistemic clarity of the source material — independent of fit_score)
  - fit_score:  1 (poor fit) to 5 (excellent fit) — how well the assigned harm + mechanism
                classes ACTUALLY FIT this path. Score INDEPENDENTLY from confidence:
                a path can be clearly-read (confidence=5) but poorly-fit by any existing
                class (fit_score=2). Low fit_score is the signal to use "unassigned" rather
                than force-fit a bad class.
  - fit_note: required when fit_score <= 3 OR when "unassigned" is used —
              1-clause justification

UNASSIGNED DISCIPLINE:
- Use "unassigned" when NO existing class fits well AND new-class proposal would
  fail MIN_GROUP_SIZE (single-paper specifics, niche framings without ≥3 batch peers).
- Unassigned is PREFERRED over force-fit when fit_score would be ≤ 2. Bad
  assignments contaminate the matrix; unassigned paths are honestly flagged for
  consolidation / final audit attention.
- You MAY have harm_class unassigned but mechanism_class assigned (or vice versa)
  — they are independent decisions.

CONTINUUM AWARENESS (REQUIRED):
- For harm_class: use the early causal chain (risk node + first 1-2 body nodes), NOT
  just the risk node. When the risk node is meta-level (e.g., "existential
  catastrophe"), prefer the OBJECT-LEVEL harm class that names the specific
  causal mechanism evident in the body.
- For mechanism_class: use the late causal chain (last 1-2 body nodes + intervention).
  Prefer the GENERAL TRANSFERABLE mechanism class over specific implementations.
  If the intervention is a project name / acronym (e.g., "Run AGISF reading group"),
  use the body to infer the underlying mechanism family.

RISK / INTERVENTION DECOUPLING (REQUIRED):
Resist the tempting shortcut of inferring the harm_class from the mechanism (e.g.,
"this is an interpretability intervention, therefore the harm = AI opacity"). The
risk node and early body nodes are the ground truth for harm_class — read them
INDEPENDENTLY of the intervention. Many interpretability interventions, for
example, target downstream harms (deception, misaligned-power-seeking) where
opacity is just the upstream enabler, not the named risk in the path.
Apply this discipline especially when the intervention is intuitively coupled to
a single canonical harm class — verify the path's specific risk-side chain
before defaulting to that coupling.

NEW-CLASS DISCIPLINE (MIN_GROUP_SIZE = {MIN_GROUP_SIZE}):
- Propose new harm or mechanism class ONLY when (a) no existing class fits AND
  (b) you expect >= {MIN_GROUP_SIZE} paths from THIS batch + reasonable corpus
  extrapolation will fit. Singleton new groups are forbidden.
- Use the same proposed class_name VERBATIM across paths that share the new class.
- Code post-processing will DROP any new class with <{MIN_GROUP_SIZE} members in
  this batch and force-fit affected paths to the closest existing class; mark
  forced-fits with low confidence + fit_note so they surface for consolidation.

AXIS VOCABULARIES — use existing values from CURRENT CATALOG. If a needed value
is genuinely missing, emit "OTHER:<free_text>" and flag the extension in
catalog_flags.axis_value_extensions.

============================================================
CATALOG IMPROVEMENT FLAGS (output at end of JSON)
============================================================

Based on what you SEE in this batch's 75 paths, flag (but do not apply) any of:
  - merge_candidates: two existing classes that appear semantically redundant
  - rename_suggestions: classes whose name/description should be tightened
  - generalize_suggestions: a class whose stated scope is NARROWER than its actual
    member set warrants. EXAMPLE: "data poisoning on distributed training" with
    several members poisoning non-distributed training too — propose broader
    name + description that subsumes the broader scope without losing precision.
  - axis_value_extensions: new enum values needed for an axis
  - split_candidates: existing classes that look heterogeneous given new evidence
  - reassign_candidates: existing-class members you judge are mis-fit and should
    move to a different class (cite path_id + suggested_class_id + rationale).
    Final audit will adjudicate.

Keep flags concise — consolidation pass will adjudicate.

============================================================
INPUT PATHS ({n} paths)
============================================================

{body}

============================================================
OUTPUT FORMAT (STRICT — validation will reject malformed responses)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.
- `assignments` array MUST contain exactly {n} entries.

Schema:

{{
  "assignments": [
    {{"path_id": "path_NNNNN_dedup",
      "harm_class":      {{"existing": "HC005"}}  OR  {{"new": {{"class_name": "...", "class_description": "...", "is_capability_gap": false}}}}  OR  {{"unassigned": true, "reason": "..."}},
      "mechanism_class": {{"existing": "MC012"}}  OR  {{"new": {{"class_name": "...", "class_description": "..."}}}}  OR  {{"unassigned": true, "reason": "..."}},
      "axis_values": {{"lifecycle_stage": "fine-tune", "modality": "LLM", "methodology": "algorithmic", "severity": "moderate", "emergence_stage": "deployment-runtime", "harm_target": "human-flourishing-rights"}},
      "harm_target_evidence": "<1 clause citing which risk-side node/description supports the harm_target value>",
      "confidence": 4,
      "fit_score": 4,
      "fit_note": "<required when fit_score <= 3 OR when unassigned>"
    }}
  ],
  "catalog_flags": {{
    "merge_candidates": [{{"a": "HC###", "b": "HC###", "rationale": "..."}}],
    "rename_suggestions": [{{"class_id": "HC###", "new_name": "...", "new_description": "...", "rationale": "..."}}],
    "generalize_suggestions": [{{"class_id": "HC###", "broader_name": "...", "broader_description": "...", "rationale": "..."}}],
    "axis_value_extensions": [{{"axis": "modality", "new_value": "biological", "rationale": "..."}}],
    "split_candidates": [{{"class_id": "HC###", "rationale": "..."}}],
    "reassign_candidates": [{{"path_id": "path_NNNNN_dedup", "current_class_id": "HC###", "suggested_class_id": "HC###", "rationale": "..."}}]
  }}
}}END_SENTINEL_{sentinel}

Produce the assignments + flags now."""


# ============================================================
# Routing resolver + MIN_GROUP_SIZE enforcement
# ============================================================
def _normalize_class_name(name):
    """Normalize a class name for fuzzy duplicate detection.

    Maps 'Audio / speech perception accuracy ceiling' and
    'Audio / speech-perception accuracy ceiling' to the same key, so two
    near-identical proposals (e.g. from different chunks of an R4 sweep)
    don't both pass the idempotent-by-name guard and create duplicate
    classes. Cycle 3 R4 sweep #13 produced HC025+HC026 as cross-chunk
    duplicates because the exact-string check missed a hyphen difference;
    this normalization closes that gap forward.

    Normalization rules:
    - lowercase
    - replace separators ([-_/.,()]) with single space
    - collapse multiple whitespace into single space
    - strip leading/trailing whitespace
    """
    import re

    t = (name or "").lower()
    t = re.sub(r"[-_/.,()]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _resolve_and_enforce(parsed, catalog, batch_idx):
    """Apply Opus's per-path assignments to the catalog.
    - Add new harm/mech classes to catalog (idempotent by NORMALIZED class_name —
      handles near-duplicate proposals differing only by punctuation/case).
    - Count members per NEW class within THIS batch; if <MIN_GROUP_SIZE, DROP
      the new class and force-fit affected paths to closest existing class
      (by Opus's confidence ranking; if no fallback indicated, mark as
      fallback_needed and append to flags log for next consolidation).
    - Returns (updated_catalog, resolved_assignments, forced_fit_count).
    """
    # Keep both exact-name lookup (for in-batch resolution against existing
    # catalog) AND normalized-name lookup (for cross-batch / cross-chunk
    # duplicate detection on NEW class proposals).
    hc_by_name = {h["class_name"]: h for h in catalog["harm_classes"]}
    mc_by_name = {m["class_name"]: m for m in catalog["mechanism_classes"]}
    hc_by_norm = {
        _normalize_class_name(h["class_name"]): h for h in catalog["harm_classes"]
    }
    mc_by_norm = {
        _normalize_class_name(m["class_name"]): m for m in catalog["mechanism_classes"]
    }
    hc_ids = {h["class_id"] for h in catalog["harm_classes"]}
    mc_ids = {m["class_id"] for m in catalog["mechanism_classes"]}
    next_hc_idx = max(
        [int(h["class_id"][2:]) for h in catalog["harm_classes"]], default=0
    )
    next_mc_idx = max(
        [int(m["class_id"][2:]) for m in catalog["mechanism_classes"]], default=0
    )

    # First pass: collect new-class proposals + assign canonical ids tentatively
    new_hc_members = defaultdict(list)  # name -> [path_id, ...]
    new_mc_members = defaultdict(list)
    proposed_hc_meta = {}  # name -> {description, is_capability_gap}
    proposed_mc_meta = {}

    for a in parsed.get("assignments", []):
        pid = a.get("path_id")
        if not pid:
            continue
        hc_field = a.get("harm_class", {})
        mc_field = a.get("mechanism_class", {})
        if isinstance(hc_field, dict) and "new" in hc_field:
            new_h = hc_field["new"]
            nm = (new_h.get("class_name") or "").strip()
            if nm:
                new_hc_members[nm].append(pid)
                proposed_hc_meta[nm] = {
                    "description": new_h.get("class_description", ""),
                    "is_capability_gap": new_h.get("is_capability_gap", False),
                }
        if isinstance(mc_field, dict) and "new" in mc_field:
            new_m = mc_field["new"]
            nm = (new_m.get("class_name") or "").strip()
            if nm:
                new_mc_members[nm].append(pid)
                proposed_mc_meta[nm] = {
                    "description": new_m.get("class_description", ""),
                }

    # Determine which new classes survive MIN_GROUP_SIZE
    surviving_new_hcs = {
        nm for nm, pids in new_hc_members.items() if len(pids) >= MIN_GROUP_SIZE
    }
    surviving_new_mcs = {
        nm for nm, pids in new_mc_members.items() if len(pids) >= MIN_GROUP_SIZE
    }
    dropped_hc_names = set(new_hc_members.keys()) - surviving_new_hcs
    dropped_mc_names = set(new_mc_members.keys()) - surviving_new_mcs

    # Append surviving new classes to catalog (idempotent by EXACT name AND
    # by NORMALIZED name). The normalized-name guard prevents cross-chunk /
    # cross-batch duplicate creation when two proposals differ only by
    # punctuation/case (e.g. 'X perception' vs 'X-perception'). When a
    # normalized-name collision is detected, redirect the proposal to the
    # existing entry's class_id by aliasing hc_by_name[nm] to it — this
    # makes the second-pass assignment loop route paths to the canonical class.
    for nm in surviving_new_hcs:
        if nm in hc_by_name:
            continue
        norm = _normalize_class_name(nm)
        if norm in hc_by_norm:
            existing = hc_by_norm[norm]
            hc_by_name[nm] = existing  # alias the new name to the existing entry
            continue
        next_hc_idx += 1
        new_id = f"HC{next_hc_idx:03d}"
        entry = {
            "class_id": new_id,
            "class_name": nm,
            "class_description": proposed_hc_meta[nm]["description"],
            "is_capability_gap": proposed_hc_meta[nm].get("is_capability_gap", False),
        }
        catalog["harm_classes"].append(entry)
        hc_by_name[nm] = entry
        hc_by_norm[norm] = entry
        hc_ids.add(new_id)
    for nm in surviving_new_mcs:
        if nm in mc_by_name:
            continue
        norm = _normalize_class_name(nm)
        if norm in mc_by_norm:
            existing = mc_by_norm[norm]
            mc_by_name[nm] = existing
            continue
        next_mc_idx += 1
        new_id = f"MC{next_mc_idx:03d}"
        entry = {
            "class_id": new_id,
            "class_name": nm,
            "class_description": proposed_mc_meta[nm]["description"],
        }
        catalog["mechanism_classes"].append(entry)
        mc_by_name[nm] = entry
        mc_by_norm[norm] = entry
        mc_ids.add(new_id)

    # Second pass: emit resolved assignments, dropping sub-min new groups
    resolved = []
    forced_fit_count = 0
    for a in parsed.get("assignments", []):
        pid = a.get("path_id")
        if not pid:
            continue
        hc_field = a.get("harm_class", {})
        mc_field = a.get("mechanism_class", {})

        # Resolve HC
        hc_id = None
        hc_status = "assigned"  # assigned | unassigned | force_fit_pending
        force_note = []
        if isinstance(hc_field, dict):
            if hc_field.get("unassigned"):
                hc_status = "unassigned"
                force_note.append(
                    f"hc_unassigned:{(hc_field.get('reason') or '')[:60]}"
                )
            elif "existing" in hc_field:
                hc_id = hc_field["existing"] if hc_field["existing"] in hc_ids else None
            elif "new" in hc_field:
                nm = (hc_field["new"].get("class_name") or "").strip()
                if nm in surviving_new_hcs:
                    hc_id = hc_by_name[nm]["class_id"]
                elif nm in dropped_hc_names:
                    hc_status = "force_fit_pending"
                    forced_fit_count += 1
                    force_note.append(f"hc_new_dropped:{nm[:40]}")
        # Resolve MC
        mc_id = None
        mc_status = "assigned"
        if isinstance(mc_field, dict):
            if mc_field.get("unassigned"):
                mc_status = "unassigned"
                force_note.append(
                    f"mc_unassigned:{(mc_field.get('reason') or '')[:60]}"
                )
            elif "existing" in mc_field:
                mc_id = mc_field["existing"] if mc_field["existing"] in mc_ids else None
            elif "new" in mc_field:
                nm = (mc_field["new"].get("class_name") or "").strip()
                if nm in surviving_new_mcs:
                    mc_id = mc_by_name[nm]["class_id"]
                elif nm in dropped_mc_names:
                    mc_status = "force_fit_pending"
                    forced_fit_count += 1
                    force_note.append(f"mc_new_dropped:{nm[:40]}")

        resolved.append(
            {
                "path_id": pid,
                "harm_class_id": hc_id,
                "harm_class_status": hc_status,
                "mechanism_class_id": mc_id,
                "mechanism_class_status": mc_status,
                "axes": a.get("axis_values", {}),
                "harm_target_evidence": a.get("harm_target_evidence"),
                "confidence": a.get("confidence"),
                "fit_score": a.get("fit_score"),
                "fit_note": "; ".join(force_note) if force_note else a.get("fit_note"),
                "source": f"routing_batch_{batch_idx:04d}",
            }
        )

    return (
        catalog,
        resolved,
        forced_fit_count,
        sorted(dropped_hc_names),
        sorted(dropped_mc_names),
    )


def _append_flags(parsed, batch_idx):
    """Append catalog_flags from a batch to phase2_routing_flags.jsonl."""
    flags = parsed.get("catalog_flags", {}) or {}
    if not any(flags.values()):
        return 0
    with open(FLAGS_LOG_FP, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "batch_idx": batch_idx,
                    "flags": flags,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    n = sum(
        len(flags.get(k, []))
        for k in [
            "merge_candidates",
            "rename_suggestions",
            "axis_value_extensions",
            "split_candidates",
        ]
    )
    return n


def _rebuild_routing_assignments_jsonl():
    """Merged assignments file from: pilot + saved routing batches + path
    reassignment override log. Idempotent — overwrites from disc sources
    every time. Per-batch JSONs are immutable; reassignments from
    consolidation/misfit_review/etc. are applied via the override log."""
    rows = []
    rows.extend(_import_pilot_assignments())
    if BATCH_DIR.exists():
        for bf in sorted(BATCH_DIR.glob("batch_*.json")):
            d = json.loads(bf.read_text(encoding="utf-8"))
            rows.extend(d.get("resolved_assignments", []))
    # Apply path reassignment overrides (e.g., from consolidation splits +
    # misfit_review). Reassignment keys can be set to None to UNASSIGN.
    overrides = _load_path_reassignments()
    if overrides:
        for r in rows:
            ov = overrides.get(r.get("path_id"))
            if ov:
                if "harm_class_id" in ov:
                    r["harm_class_id"] = ov["harm_class_id"]
                    r["harm_class_status"] = (
                        "assigned" if ov["harm_class_id"] else "unassigned"
                    )
                if "mechanism_class_id" in ov:
                    r["mechanism_class_id"] = ov["mechanism_class_id"]
                    r["mechanism_class_status"] = (
                        "assigned" if ov["mechanism_class_id"] else "unassigned"
                    )
                r.setdefault("history", []).append(
                    {
                        "edit": f"reassigned via {ov.get('source', '?')}",
                        "rationale": ov.get("rationale", "")[:160],
                    }
                )
    # Atomic write
    tmp = ASSIGNMENTS_FP.with_suffix(
        ASSIGNMENTS_FP.suffix + f".tmp.{uuid.uuid4().hex[:6]}"
    )
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, ASSIGNMENTS_FP)
    return len(rows)


def _compute_class_counts():
    """Counts per HC + MC from current merged assignments jsonl."""
    hc_counts = Counter()
    mc_counts = Counter()
    if not ASSIGNMENTS_FP.exists():
        return hc_counts, mc_counts
    for line in ASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("harm_class_id"):
            hc_counts[r["harm_class_id"]] += 1
        if r.get("mechanism_class_id"):
            mc_counts[r["mechanism_class_id"]] += 1
    return hc_counts, mc_counts


# ============================================================
# Consolidation prompt + runner
# ============================================================
def make_consolidation_prompt(
    catalog, accumulated_flags, hc_counts, mc_counts, sentinel
):
    """Single Opus call processes accumulated catalog-improvement flags."""
    catalog_str = _fmt_catalog_for_routing(catalog, hc_counts, mc_counts)
    flags_str = json.dumps(accumulated_flags, indent=2, ensure_ascii=False)[:30000]
    watch_str = _load_watch_items() or "(no watch items recorded yet)"
    sub_channels_str = _load_surfaced_sub_channels_for_prompt()
    return f"""You are consolidating the doublet catalog after a routing-batch cycle.

{M.PAPER_DELIVERABLE_CONTEXT}

============================================================
NO-HETEROGENEOUS-CATCH-ALL POLICY (HARD RULE — also applies here)
============================================================

The trigger for splitting is HETEROGENEITY, not size. Large coherent classes
(n=50+ on one mechanism family) are FULLY ACCEPTABLE — they ARE the paper's
headline finding. A class is REQUIRED to split when it bundles ≥2 distinct
sub-mechanism families with ≥3 members each. Forbidden defense: "broad-but-
coherent" / "general enough to cover diverse paths" — that IS the catch-all
anti-pattern.

`unassigned` is the only legitimate residual bucket. Never accept a forced fit
to a heterogeneous class.

============================================================
ACTIVE WATCH-ITEMS (consult and update where adjudication closes them)
============================================================

{watch_str}

============================================================
SURFACED SUB-CHANNELS WATCHLIST (cumulative across prior misfit_reviews + consolidations)

These sub-channels were named by prior reviews as catalog gaps but did not yet
reach the >=3-peer quorum. Adjudicate each in light of the accumulated flags
below:
- If a flag's `member_path_ids_to_move` brings a watchlist sub-channel to >=3
  cumulative peers, you SHOULD APPLY_SPLIT now with the merged peer list.
- If a flag aligns with a watchlist sub-channel but still short of 3 peers,
  add this batch's contribution to the sub-channel via `surfaced_sub_channels`
  output (peer accumulation across cycles).
- Defended-homogeneous decisions still apply: heterogeneity (≥2 sub-families
  with ≥3 each), not size, triggers required split.
============================================================

{sub_channels_str}

============================================================
CURRENT CATALOG
============================================================

{catalog_str}

============================================================
ACCUMULATED CATALOG-IMPROVEMENT FLAGS FROM RECENT ROUTING BATCHES
============================================================

{flags_str}

============================================================
YOUR TASK — adjudicate flags + APPLY splits when warranted
============================================================

For each flag category:

  - merge_candidates: MERGE (apply remap) | KEEP_DISTINCT (with rationale)

  - rename_suggestions: APPLY (with new_name + new_description) | REJECT

  - generalize_suggestions: APPLY (with broader_name + broader_description)
    | REJECT — apply only when broader scope is supported by actual member set
    diversity AND does not blur the boundary to an adjacent class

  - axis_value_extensions: ACCEPT (add to controlled vocab) | REJECT

  - split_candidates: ONE OF
      (a) APPLY_SPLIT — emit splits_applied entries WHEN the flag rationale
          already names ≥3 in-batch peer paths sharing a coherent sub-channel.
          For each split, allocate a NEW class with new_class_name +
          new_class_description, and list member_path_ids_to_move (the ≥3
          paths to migrate from the from_class_id to the new class).
          THIS IS THE PREFERRED ACTION when flag rationale provides member
          path_ids.
      (b) SCHEDULE_DEEP_DIVE — for cases where rationale describes the split
          but doesn't name specific member paths (deferred to final_misfit_sweep).
      (c) DROP — flag is unjustified (defend the class as homogeneous in
          rationale).

  - reassign_candidates: APPLY (emit reassignments_applied with path_id +
    new harm_class_id and/or mechanism_class_id) | REJECT — apply only when
    the path's chain clearly fits suggested better.

Also: identify any classes currently with count=1 that you judge are GENUINELY
NICHE (keep) vs LIKELY FRAGMENT (mark for force-merge in final audit).

============================================================
OUTPUT FORMAT (STRICT)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.

Schema:

{{
  "merges_applied": [{{"from": "HC###", "to": "HC###", "rationale": "..."}}],
  "renames_applied": [{{"class_id": "HC###", "new_name": "...", "new_description": "...", "rationale": "..."}}],
  "generalizations_applied": [{{"class_id": "HC###", "broader_name": "...", "broader_description": "...", "rationale": "..."}}],
  "axis_extensions_applied": [{{"axis": "modality", "new_value": "biological", "rationale": "..."}}],
  "splits_applied": [
    {{"from_class_id": "HC###",
      "new_class_name": "...",
      "new_class_description": "...",
      "is_capability_gap": false,
      "member_path_ids_to_move": ["path_NNNNN_dedup", "path_NNNNN_dedup", "path_NNNNN_dedup"],
      "rationale": "..."}}
  ],
  "split_dives_scheduled": [{{"class_id": "HC###", "rationale": "..."}}],
  "reassignments_applied": [{{"path_id": "path_NNNNN_dedup", "from_class_id": "HC###", "to_class_id": "HC###", "side": "harm" | "mechanism", "rationale": "..."}}],
  "singletons_review": [{{"class_id": "HC###", "verdict": "keep_niche" | "force_merge_at_final"}}],
  "defended_homogeneous_classes": [{{"class_id": "HC###", "rationale": "explicit argument that this class is one tight mechanism family despite size"}}],
  "surfaced_sub_channels": [
    {{"name": "<short kebab-or-prose name for the gap>",
      "side": "HC" | "MC",
      "rationale": "<1-2 sentences>",
      "named_peer_path_ids": ["path_NNNNN_dedup", "..."],
      "n_peers_in_corpus_so_far": <int>,
      "min_required": 3,
      "confirmed_out_of_scope": <true if this sub-channel falls outside the AI-safety paper scope (non-AI-safety risk/mechanism family) and should remain UNASSIGNED even if peers reach min_required; false otherwise>}}
  ],
  "summary": "<3-5 sentences on this consolidation pass>"
}}END_SENTINEL_{sentinel}

NOTE on `surfaced_sub_channels`: if any flag in the input names a coherent
catalog gap that still lacks >=3 peers (so APPLY_SPLIT is not yet warranted),
emit it here so future cycles can accumulate peers. If the watchlist above
shows the sub-channel was already named in a prior review, ADD this cycle's
new peer paths to `named_peer_path_ids` — the system dedupes across sources.
When cumulative peers reach >=3, emit APPLY_SPLIT in `splits_applied` with
the full peer list instead.

IMPORTANT — `confirmed_out_of_scope` flag: set this to true for any
sub-channel that is non-AI-safety (e.g. AI applied to non-AI cybersecurity,
non-AI cryptography, classical control engineering, conservation AI,
human cognitive training, corporate finance, generic ML productivity tooling,
non-AI existential cause areas). For OOS sub-channels we DO NOT want to
propose a new class even at quorum — paths matching them stay UNASSIGNED.
Setting this flag prevents the sub-channel from re-surfacing as an
actionable new-class candidate in future review cycles.

Produce the consolidation now."""


def make_final_audit_prompt(catalog, hc_counts, mc_counts, total_assigned, sentinel):
    catalog_str = _fmt_catalog_for_routing(catalog, hc_counts, mc_counts)
    watch_str = _load_watch_items() or "(no watch items recorded yet)"
    sub_channels_str = _load_surfaced_sub_channels_for_prompt()
    return f"""You are doing the END-OF-RUN final audit of the doublet catalog before
downstream analysis.

{M.PAPER_DELIVERABLE_CONTEXT}

Total paths routed: {total_assigned}.

============================================================
ACTIVE WATCH-ITEMS (full history — every routing batch, sweep, consolidation, misfit_review)
============================================================

{watch_str}

============================================================
SURFACED SUB-CHANNELS WATCHLIST (catalog gaps named but not yet split)

Each entry below is a sub-channel a prior review identified but lacked >=3
named peers to act on. For final audit, you should:
  (a) For any entry at >=3 peers — APPLY the split now (emit in splits_applied
      schema slot) since this is the last chance before downstream analysis.
  (b) For any entry with 1-2 peers that NO LONGER fits the corpus — explicitly
      retire it (note in summary) so future analysis isn't misled by stale
      gaps.
  (c) For entries that are still legitimate gaps but lack quorum — note in
      summary as "publication caveat: catalog has known under-served sub-
      channels at N=<count>".
============================================================

{sub_channels_str}

============================================================
CURRENT CATALOG (post-consolidations)
============================================================

{catalog_str}

============================================================
YOUR TASK
============================================================

1. SINGLETON CLEANUP — for any HC or MC with count=1 or 2 still present, decide:
   FORCE_MERGE (give target class_id), KEEP_NICHE (with rationale), or
   FORCE_RENAME (if the singleton is generic-named but actually distinct).

2. CLASS DEFINITION TIGHTENING — for any class whose name OR description is
   imprecise given its actual membership distribution, propose tightened
   name + description.

3. FINAL AXES VOCABULARY — for each axis, list the values that actually
   appeared (used) vs the controlled vocab. Propose vocab pruning (drop unused
   values) or vocab additions (consolidate OTHER:* free-text values that
   recurred).

4. PAPER-DELIVERABLE READINESS — note which HCs and MCs are best-populated
   transferability examples (deliverable 2), which HCs are under-served (only
   1-2 MC-types observed → deliverable 3 candidates), and any structural
   concerns about the headline harm_class × mechanism_class matrix.

============================================================
OUTPUT FORMAT (STRICT)
============================================================

- Output ONLY one JSON object. No preamble.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.

Schema:

{{
  "singleton_decisions": [{{"class_id": "HC###", "verdict": "FORCE_MERGE" | "KEEP_NICHE" | "FORCE_RENAME", "target_class_id": "<if merge>", "new_name": "<if rename>", "new_description": "<if rename>", "rationale": "..."}}],
  "class_tightenings": [{{"class_id": "HC###", "new_name": "...", "new_description": "...", "rationale": "..."}}],
  "axis_vocab_final": [{{"axis": "lifecycle_stage", "values_used": [...], "values_pruned": [...], "values_added_from_OTHER": [...]}}],
  "transferability_examples": [{{"class_id": "MC###", "n_harm_classes_touched": N, "harm_class_ids": [...]}}],
  "gap_candidates": [{{"class_id": "HC###", "n_mech_classes_observed": N, "rationale": "..."}}],
  "matrix_concerns": "<2-4 sentences>",
  "summary": "<3-5 sentences on the final catalog state>"
}}END_SENTINEL_{sentinel}

Produce the final audit now."""


# ============================================================
# Runners
# ============================================================
def run_opus_routing(n_batches=None, batch_size=BATCH_SIZE):
    """Main routing loop."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
    catalog = _load_active_catalog()
    state = _load_state()

    # Bootstrap pilot assignments into merged jsonl on first run
    if not state.get("pilot_assignments_seeded"):
        n_rows = _rebuild_routing_assignments_jsonl()
        state["pilot_assignments_seeded"] = True
        _save_state(state)
        print(
            f"  seeded routing assignments jsonl: {n_rows} rows from pilot", flush=True
        )

    print("loading deduped paths ...", flush=True)
    paths = _load_deduped_paths()
    print(f"  {len(paths)} deduped paths", flush=True)
    print("loading node_attrs ...", flush=True)
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    print(f"  {len(na)} nodes", flush=True)

    done = _get_done_path_ids()
    remaining = [p for p in paths if p["path_id"] not in done]
    print(f"already routed: {len(done)}; remaining: {len(remaining)}", flush=True)
    if not remaining:
        print("All paths routed. Run --mode final_audit next.", flush=True)
        return

    rng = random.Random(ROUTING_RNG_SEED)
    rng.shuffle(remaining)

    existing_batches = sorted(BATCH_DIR.glob("batch_*.json"))
    next_batch_idx = (
        max(
            [int(re.search(r"batch_(\d+)", f.name).group(1)) for f in existing_batches],
            default=-1,
        )
    ) + 1
    n_remaining_total = (len(remaining) + batch_size - 1) // batch_size
    n_to_run = (
        n_remaining_total if n_batches is None else min(n_batches, n_remaining_total)
    )
    print(
        f"will run {n_to_run} batches starting at batch_{next_batch_idx:04d}",
        flush=True,
    )

    for bi in range(n_to_run):
        batch_idx = next_batch_idx + bi
        batch = remaining[bi * batch_size : (bi + 1) * batch_size]
        hc_counts, mc_counts = _compute_class_counts()
        print(
            f"\n--- routing_batch_{batch_idx:04d}  paths={len(batch)}  "
            f"catalog={len(catalog['harm_classes'])} HC + "
            f"{len(catalog['mechanism_classes'])} MC ---",
            flush=True,
        )
        sentinel = uuid.uuid4().hex[:12]
        prompt = make_routing_prompt(batch, catalog, hc_counts, mc_counts, na, sentinel)
        print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)

        json_part, dur, _, err = M.streaming_call_with_validation(
            prompt,
            sentinel,
            f"routing_batch_{batch_idx:04d}",
            PARTIAL_FP,
            model="claude-opus-4-7",
        )
        if err or not json_part:
            print(f"  BATCH FAILED ({err}); skipping; partial preserved", flush=True)
            M._preserve_failed_partial(
                PARTIAL_FP, batch_idx, reason=f"routing_stream_err={err}"
            )
            continue
        try:
            parsed, method = _robust_json_parse(json_part, '{"assignments":[')
            if method != "direct":
                print(f"  RECOVERED via {method}", flush=True)
        except json.JSONDecodeError as e:
            print(f"  JSON parse unrecoverable: {e}", flush=True)
            M._preserve_failed_partial(
                PARTIAL_FP, batch_idx, reason=f"routing_unrecoverable={e}"
            )
            continue

        catalog, resolved, forced, dropped_hc, dropped_mc = _resolve_and_enforce(
            parsed, catalog, batch_idx
        )
        n_flagged = _append_flags(parsed, batch_idx)
        _save_active_catalog(catalog, kind_suffix=f"routing_batch_{batch_idx:04d}")

        batch_out = BATCH_DIR / f"batch_{batch_idx:04d}.json"
        M.atomic_write_json(
            batch_out,
            {
                "batch_idx": batch_idx,
                "model": "claude-opus-4-7",
                "n_input_paths": len(batch),
                "duration_sec": dur,
                "assignments": parsed.get("assignments", []),
                "resolved_assignments": resolved,
                "catalog_flags": parsed.get("catalog_flags", {}),
                "dropped_new_hc_names": dropped_hc,
                "dropped_new_mc_names": dropped_mc,
                "forced_fit_count": forced,
            },
        )

        n_rows = _rebuild_routing_assignments_jsonl()
        _append_step_log(
            step_type="routing_batch",
            summary=f"batch_{batch_idx:04d}: {len(resolved)} resolved, "
            f"{forced} forced-fits, {n_flagged} flags emitted",
            extra={
                "batch_idx": batch_idx,
                "n_input_paths": len(batch),
                "n_resolved": len(resolved),
                "forced_fits": forced,
                "dropped_hc_proposals": len(dropped_hc),
                "dropped_mc_proposals": len(dropped_mc),
                "n_flagged": n_flagged,
                "duration_sec": dur,
            },
        )
        print(
            f"  -> {len(resolved)} resolved, {forced} forced-fits "
            f"({len(dropped_hc)} HC + {len(dropped_mc)} MC dropped sub-min); "
            f"{n_flagged} flags; merged jsonl now {n_rows} rows",
            flush=True,
        )

        # Auto-rebuild xlsx after each batch (graceful on PermissionError if Excel open)
        try:
            from phase2_routing_to_xlsx import build_routing_xlsx

            res = build_routing_xlsx()
            print(
                f"  xlsx refreshed: {res['n_paths']} paths, {res['n_harm_classes']} HC + "
                f"{res['n_mech_classes']} MC",
                flush=True,
            )
        except PermissionError as e:
            print(f"  xlsx WRITE BLOCKED (Excel open?): {e}", flush=True)
        except Exception as e:
            print(
                f"  xlsx build FAILED ({type(e).__name__}): {e}; continuing", flush=True
            )

        state["total_batches_run"] += 1
        _save_state(state)

    print("\n=== Opus routing checkpoint complete ===", flush=True)
    print(f"  total batches so far: {state['total_batches_run']}", flush=True)
    since_consol = state["total_batches_run"] - max(
        0, state["last_consolidation_at_batch_idx"] + 1
    )
    if since_consol >= CONSOLIDATION_EVERY:
        print(
            f"  >>> {since_consol} batches since last consolidation; run "
            f"`--mode consolidation` next.",
            flush=True,
        )


def run_opus_routing_parallel(n_batches=None, n_workers=5, batch_size=BATCH_SIZE):
    """Parallel routing: spawn n_workers concurrent batches per group, then
    serial-merge in batch_idx order. Each worker streams to its own partial
    file. Catalog snapshot is shared across workers in a group; new-class
    proposals are merged serially after all workers complete (idempotent by
    class_name via _resolve_and_enforce, so duplicate proposals collapse).
    """
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
    catalog = _load_active_catalog()
    state = _load_state()

    if not state.get("pilot_assignments_seeded"):
        n_rows = _rebuild_routing_assignments_jsonl()
        state["pilot_assignments_seeded"] = True
        _save_state(state)
        print(
            f"  seeded routing assignments jsonl: {n_rows} rows from pilot", flush=True
        )

    print("loading deduped paths ...", flush=True)
    paths = _load_deduped_paths()
    print(f"  {len(paths)} deduped paths", flush=True)
    print("loading node_attrs ...", flush=True)
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    print(f"  {len(na)} nodes", flush=True)

    done = _get_done_path_ids()
    remaining = [p for p in paths if p["path_id"] not in done]
    print(f"already routed: {len(done)}; remaining: {len(remaining)}", flush=True)
    if not remaining:
        print("All paths routed. Run --mode final_audit next.", flush=True)
        return

    rng = random.Random(ROUTING_RNG_SEED)
    rng.shuffle(remaining)

    existing_batches = sorted(BATCH_DIR.glob("batch_*.json"))
    next_batch_idx = (
        max(
            [int(re.search(r"batch_(\d+)", f.name).group(1)) for f in existing_batches],
            default=-1,
        )
    ) + 1
    n_remaining_total = (len(remaining) + batch_size - 1) // batch_size
    n_to_run = (
        n_remaining_total if n_batches is None else min(n_batches, n_remaining_total)
    )
    print(
        f"will run {n_to_run} batches in groups of {n_workers}, "
        f"starting at batch_{next_batch_idx:04d}",
        flush=True,
    )

    batches_completed = 0
    while batches_completed < n_to_run:
        group_start = batches_completed
        group_size = min(n_workers, n_to_run - batches_completed)
        group_batch_idxs = [next_batch_idx + group_start + i for i in range(group_size)]
        group_path_slices = [
            remaining[
                (group_start + i) * batch_size : (group_start + i + 1) * batch_size
            ]
            for i in range(group_size)
        ]
        hc_counts, mc_counts = _compute_class_counts()

        worker_args = []
        for wi in range(group_size):
            bi = group_batch_idxs[wi]
            batch = group_path_slices[wi]
            if not batch:
                continue
            sentinel = uuid.uuid4().hex[:12]
            prompt = make_routing_prompt(
                batch, catalog, hc_counts, mc_counts, na, sentinel
            )
            partial = STEP1 / f"phase2_routing_partial_w{wi}.txt"
            worker_args.append(
                {
                    "wi": wi,
                    "batch_idx": bi,
                    "batch": batch,
                    "sentinel": sentinel,
                    "prompt": prompt,
                    "partial": partial,
                }
            )

        if not worker_args:
            break

        print(
            f"\n=== Parallel group: {len(worker_args)} batches in flight "
            f"(batch_{group_batch_idxs[0]:04d}..batch_{group_batch_idxs[-1]:04d}) ===",
            flush=True,
        )
        for wa in worker_args:
            print(
                f"  worker {wa['wi']}: batch_{wa['batch_idx']:04d}, "
                f"{len(wa['batch'])} paths, prompt {len(wa['prompt'])} chars",
                flush=True,
            )

        # Run all workers in parallel via ThreadPoolExecutor.
        # Each thread spawns its own `claude -p` subprocess with isolated partial path.
        results = {}
        t_group_start = time.time()
        with ThreadPoolExecutor(max_workers=len(worker_args)) as ex:
            futures = {}
            for wa in worker_args:
                fut = ex.submit(
                    M.streaming_call_with_validation,
                    wa["prompt"],
                    wa["sentinel"],
                    f"routing_batch_{wa['batch_idx']:04d}",
                    wa["partial"],
                )
                futures[fut] = wa
            for fut in as_completed(futures):
                wa = futures[fut]
                try:
                    json_part, dur, _, err = fut.result()
                except Exception as e:
                    json_part, dur, err = (
                        None,
                        0.0,
                        f"thread_exception: {type(e).__name__}: {str(e)[:200]}",
                    )
                results[wa["wi"]] = (json_part, dur, err, wa)
                tag = "OK" if (json_part and not err) else f"FAIL ({err})"
                print(
                    f"  worker {wa['wi']} batch_{wa['batch_idx']:04d}: {tag} "
                    f"in {dur:.0f}s",
                    flush=True,
                )
        group_wall = time.time() - t_group_start
        print(f"  group wall-clock: {group_wall:.0f}s", flush=True)

        # Serial merge in batch-idx order so catalog mutations are deterministic
        applied = 0
        skipped = 0
        for wi in sorted(results.keys()):
            json_part, dur, err, wa = results[wi]
            batch_idx = wa["batch_idx"]
            if err or not json_part:
                print(
                    f"  batch_{batch_idx:04d} FAILED: {err}; preserving partial",
                    flush=True,
                )
                M._preserve_failed_partial(
                    wa["partial"], batch_idx, reason=f"routing_stream_err={err}"
                )
                skipped += 1
                continue
            try:
                parsed, method = _robust_json_parse(json_part, '{"assignments":[')
                if method != "direct":
                    print(f"  batch_{batch_idx:04d} RECOVERED via {method}", flush=True)
            except json.JSONDecodeError as e:
                print(
                    f"  batch_{batch_idx:04d} JSON parse unrecoverable: {e}", flush=True
                )
                M._preserve_failed_partial(
                    wa["partial"], batch_idx, reason=f"routing_unrecoverable={e}"
                )
                skipped += 1
                continue

            catalog, resolved, forced, dropped_hc, dropped_mc = _resolve_and_enforce(
                parsed, catalog, batch_idx
            )
            n_flagged = _append_flags(parsed, batch_idx)
            _save_active_catalog(catalog, kind_suffix=f"routing_batch_{batch_idx:04d}")

            batch_out = BATCH_DIR / f"batch_{batch_idx:04d}.json"
            M.atomic_write_json(
                batch_out,
                {
                    "batch_idx": batch_idx,
                    "model": "claude-opus-4-7",
                    "n_input_paths": len(wa["batch"]),
                    "duration_sec": dur,
                    "assignments": parsed.get("assignments", []),
                    "resolved_assignments": resolved,
                    "catalog_flags": parsed.get("catalog_flags", {}),
                    "dropped_new_hc_names": dropped_hc,
                    "dropped_new_mc_names": dropped_mc,
                    "forced_fit_count": forced,
                    "parallel_worker_idx": wi,
                },
            )
            _append_step_log(
                step_type="routing_batch_parallel",
                summary=f"batch_{batch_idx:04d} (parallel w{wi}): {len(resolved)} resolved, "
                f"{forced} forced-fits, {n_flagged} flags emitted",
                extra={
                    "batch_idx": batch_idx,
                    "worker_idx": wi,
                    "n_input_paths": len(wa["batch"]),
                    "n_resolved": len(resolved),
                    "forced_fits": forced,
                    "dropped_hc_proposals": len(dropped_hc),
                    "dropped_mc_proposals": len(dropped_mc),
                    "n_flagged": n_flagged,
                    "duration_sec": dur,
                },
            )
            print(
                f"  batch_{batch_idx:04d} merged: {len(resolved)} resolved, "
                f"{forced} forced, {n_flagged} flags "
                f"({len(dropped_hc)} HC + {len(dropped_mc)} MC dropped sub-min)",
                flush=True,
            )
            state["total_batches_run"] += 1
            applied += 1

        # Rebuild jsonl + xlsx ONCE per group
        n_rows = _rebuild_routing_assignments_jsonl()
        print(f"  merged jsonl: {n_rows} rows", flush=True)
        try:
            from phase2_routing_to_xlsx import build_routing_xlsx

            res = build_routing_xlsx()
            print(
                f"  xlsx refreshed: {res['n_paths']} paths, "
                f"{res['n_harm_classes']} HC + {res['n_mech_classes']} MC",
                flush=True,
            )
        except PermissionError as e:
            print(f"  xlsx WRITE BLOCKED (Excel open?): {e}", flush=True)
        except Exception as e:
            print(
                f"  xlsx build FAILED ({type(e).__name__}): {e}; continuing", flush=True
            )
        _save_state(state)

        print(f"  group summary: {applied} applied, {skipped} failed", flush=True)
        # If all failed, abort the loop to avoid infinite retry on systemic failure
        if applied == 0 and skipped > 0:
            print(
                "\n  ALL WORKERS IN GROUP FAILED — aborting parallel loop. "
                "Investigate before resuming.",
                flush=True,
            )
            break

        batches_completed += group_size

    print("\n=== Parallel opus routing checkpoint complete ===", flush=True)
    print(f"  total batches so far: {state['total_batches_run']}", flush=True)


def run_consolidation():
    catalog = _load_active_catalog()
    state = _load_state()
    if not FLAGS_LOG_FP.exists():
        print("No flags log. Run routing first.", flush=True)
        return
    accumulated = []
    for line in FLAGS_LOG_FP.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            accumulated.append(json.loads(line))
    # Only flags from batches NOT YET consolidated.
    # Semantics: last_consolidation_at_batch_idx stores the batch_idx threshold
    # ABOVE WHICH (inclusive) batches still need consolidation — i.e. the
    # value is "the FIRST batch_idx not yet consolidated minus 1". A prior bug
    # used strict `>` here, which silently dropped the flag entries from the
    # batch whose index equalled the cutoff (off-by-one). Fixed 2026-05-20 to
    # `>=` after consolidation_002 wrote cutoff=15 to mean "batch 15 and up
    # remain", but the strict `>` filter would have excluded batch 15.
    cutoff = state.get("last_consolidation_at_batch_idx", -1)
    accumulated = [e for e in accumulated if e["batch_idx"] >= cutoff]
    if not accumulated:
        print(f"No new flags since batch_{cutoff:04d}.", flush=True)
        return
    print(
        f"consolidating {len(accumulated)} flag entries (batches "
        f"{cutoff + 1} .. {state['total_batches_run']}) ...",
        flush=True,
    )
    hc_counts, mc_counts = _compute_class_counts()
    consolidation_idx = state.get("total_consolidations_run", 0) + 1
    sentinel = uuid.uuid4().hex[:12]
    prompt = make_consolidation_prompt(
        catalog, accumulated, hc_counts, mc_counts, sentinel
    )
    print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)
    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        f"consolidation_{consolidation_idx:03d}",
        PARTIAL_FP,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(f"CONSOLIDATION FAILED ({err})", flush=True)
        return
    try:
        parsed, method = _robust_json_parse(json_part, '{"merges_applied":')
        if method != "direct":
            print(f"  RECOVERED via {method}", flush=True)
    except json.JSONDecodeError as e:
        print(f"JSON parse unrecoverable: {e}", flush=True)
        return

    # Save raw consolidation output
    out = CONSOLIDATION_DIR / f"consolidation_{consolidation_idx:03d}.json"
    M.atomic_write_json(
        out,
        {
            "consolidation_idx": consolidation_idx,
            "duration_sec": dur,
            "n_batches_consolidated": len(accumulated),
            "batches_range": [cutoff + 1, state["total_batches_run"]],
            "raw_output": parsed,
        },
    )

    # Apply: merges -> group_remap; renames -> in-place edit; axis extensions
    # -> add to controlled vocab; splits -> next-batch attention only
    n_merges = len(parsed.get("merges_applied", []))
    n_renames = len(parsed.get("renames_applied", []))
    n_axis_ext = len(parsed.get("axis_extensions_applied", []))

    # Apply renames in-place on catalog
    hc_by_id = {h["class_id"]: h for h in catalog["harm_classes"]}
    mc_by_id = {m["class_id"]: m for m in catalog["mechanism_classes"]}
    for ren in parsed.get("renames_applied", []):
        gid = ren.get("class_id", "")
        target = hc_by_id.get(gid) or mc_by_id.get(gid)
        if target:
            if ren.get("new_name"):
                target["class_name"] = ren["new_name"]
            if ren.get("new_description"):
                target["class_description"] = ren["new_description"]

    # Apply merges: drop merged class from catalog + write remap entry
    remap_hc = {}
    remap_mc = {}
    for mrg in parsed.get("merges_applied", []):
        src = mrg.get("from")
        dst = mrg.get("to")
        if src in hc_by_id and dst in hc_by_id and src != dst:
            remap_hc[src] = dst
        elif src in mc_by_id and dst in mc_by_id and src != dst:
            remap_mc[src] = dst
    catalog["harm_classes"] = [
        h for h in catalog["harm_classes"] if h["class_id"] not in remap_hc
    ]
    catalog["mechanism_classes"] = [
        m for m in catalog["mechanism_classes"] if m["class_id"] not in remap_mc
    ]

    # Apply axis extensions
    axes_by_name = {a["axis_name"]: a for a in catalog["axes"]}
    for ext in parsed.get("axis_extensions_applied", []):
        ax = axes_by_name.get(ext.get("axis", ""))
        new_val = ext.get("new_value", "")
        if ax and new_val and new_val not in ax.get("values", []):
            ax["values"].append(new_val)

    # Apply splits: allocate new class IDs, add to catalog, append per-path
    # reassignments to the override log. Each split must move ≥ MIN_GROUP_SIZE
    # members or it's dropped (member quorum guard).
    splits_applied_count = 0
    splits_dropped = []
    for split in parsed.get("splits_applied", []):
        from_id = split.get("from_class_id", "")
        new_name = split.get("new_class_name", "").strip()
        new_desc = split.get("new_class_description", "").strip()
        members = split.get("member_path_ids_to_move", []) or []
        rationale = split.get("rationale", "")
        if not new_name or not new_desc:
            splits_dropped.append((from_id, "missing name/description"))
            continue
        if len(members) < MIN_GROUP_SIZE:
            splits_dropped.append(
                (from_id, f"only {len(members)} members (<{MIN_GROUP_SIZE})")
            )
            continue
        kind = from_id[:2]
        if kind not in ("HC", "MC"):
            splits_dropped.append((from_id, "from_class_id not HC/MC"))
            continue
        # Allocate new id
        new_id = _next_class_id(catalog, kind)
        new_entry = {
            "class_id": new_id,
            "class_name": new_name,
            "class_description": new_desc,
            "source": f"consolidation_{consolidation_idx:03d}_split_from_{from_id}",
        }
        if kind == "HC":
            new_entry["is_capability_gap"] = bool(split.get("is_capability_gap", False))
            catalog["harm_classes"].append(new_entry)
        else:
            catalog["mechanism_classes"].append(new_entry)
        # Append reassignment record for each member
        for pid in members:
            kw = {"new_hc": new_id} if kind == "HC" else {"new_mc": new_id}
            _append_path_reassignment(
                pid,
                source=f"consolidation_{consolidation_idx:03d}_split",
                rationale=f"split-out from {from_id}: {rationale[:200]}",
                **kw,
            )
        splits_applied_count += 1
        print(
            f"  applied split: {from_id} -> {new_id} ({new_name[:60]}), "
            f"{len(members)} members moved",
            flush=True,
        )
    for drop in splits_dropped:
        print(f"  DROPPED split: {drop[0]} ({drop[1]})", flush=True)

    # Apply reassignments_applied — path-level moves (incl. UNASSIGN sentinels)
    reassigns_applied = 0
    UNASSIGN_TOKENS = {"UNASSIGNED", "UNASSIGN", "NONE", "NULL", ""}
    for ra in parsed.get("reassignments_applied", []):
        pid = ra.get("path_id", "")
        to_id = ra.get("to_class_id", "")
        side = (ra.get("side") or "").lower()
        if not pid:
            continue
        # Detect unassign signals: literal None, or string "UNASSIGNED"/"NONE"/etc.
        is_unassign = to_id is None or (
            isinstance(to_id, str) and to_id.upper().strip() in UNASSIGN_TOKENS
        )
        if is_unassign:
            # Need side to know which axis to unassign
            if side not in ("harm", "mechanism"):
                # Try to infer from from_class_id prefix
                from_id = ra.get("from_class_id", "")
                if isinstance(from_id, str) and from_id.startswith("HC"):
                    side = "harm"
                elif isinstance(from_id, str) and from_id.startswith("MC"):
                    side = "mechanism"
            if side == "harm":
                _append_path_reassignment(
                    pid,
                    source=f"consolidation_{consolidation_idx:03d}_unassign",
                    new_hc=_SENTINEL_UNASSIGN,
                    rationale=ra.get("rationale", ""),
                )
                reassigns_applied += 1
            elif side == "mechanism":
                _append_path_reassignment(
                    pid,
                    source=f"consolidation_{consolidation_idx:03d}_unassign",
                    new_mc=_SENTINEL_UNASSIGN,
                    rationale=ra.get("rationale", ""),
                )
                reassigns_applied += 1
            continue
        # Standard reassignment to a real class id
        if side not in ("harm", "mechanism"):
            side = (
                "harm"
                if (isinstance(to_id, str) and to_id.startswith("HC"))
                else (
                    "mechanism"
                    if (isinstance(to_id, str) and to_id.startswith("MC"))
                    else ""
                )
            )
        if side not in ("harm", "mechanism"):
            continue
        kw = {"new_hc": to_id} if side == "harm" else {"new_mc": to_id}
        _append_path_reassignment(
            pid,
            source=f"consolidation_{consolidation_idx:03d}_reassign",
            rationale=ra.get("rationale", ""),
            **kw,
        )
        reassigns_applied += 1

    _save_active_catalog(catalog, kind_suffix=f"consolidation_{consolidation_idx:03d}")

    # Persist remap (for jsonl rebuild)
    remap_fp = STEP1 / "phase2_routing_class_remap.json"
    if remap_fp.exists():
        cur = json.loads(remap_fp.read_text(encoding="utf-8"))
    else:
        cur = {"hc": {}, "mc": {}}
    cur.setdefault("hc", {}).update(remap_hc)
    cur.setdefault("mc", {}).update(remap_mc)
    M.atomic_write_json(remap_fp, cur)

    state["last_consolidation_at_batch_idx"] = state["total_batches_run"]
    state["total_consolidations_run"] = consolidation_idx
    _save_state(state)

    # Rebuild merged jsonl so split/reassign overrides take effect
    if splits_applied_count or reassigns_applied or remap_hc or remap_mc:
        n_rows = _rebuild_routing_assignments_jsonl()
        print(
            f"  rebuilt merged jsonl: {n_rows} rows after applying overrides",
            flush=True,
        )
        # Refresh xlsx
        try:
            from phase2_routing_to_xlsx import build_routing_xlsx

            res = build_routing_xlsx()
            print(
                f"  xlsx refreshed: {res['n_paths']} paths, "
                f"{res['n_harm_classes']} HC + {res['n_mech_classes']} MC",
                flush=True,
            )
        except Exception as e:
            print(f"  xlsx rebuild warning: {e}", flush=True)

    # Auto-append consolidation summary + applied actions to watch_items
    extra = []
    applied_summary = (
        f"**Applied:** {n_merges} merges, {n_renames} renames, "
        f"{n_axis_ext} axis vocab extensions, "
        f"{splits_applied_count} splits, {reassigns_applied} reassignments"
    )
    extra.append(applied_summary)
    if splits_dropped:
        extra.append(
            f"**Splits DROPPED ({len(splits_dropped)} due to quorum/format):** "
            f"{[(d[0], d[1]) for d in splits_dropped[:6]]}"
        )
    if parsed.get("split_dives_scheduled"):
        extra.append("**Split deep-dives scheduled (final_misfit_sweep):**")
        for s in parsed.get("split_dives_scheduled", []):
            extra.append(f"- {s.get('class_id')}: {s.get('rationale', '')[:140]}")
    if parsed.get("defended_homogeneous_classes"):
        extra.append("**Defended as homogeneous (not split):**")
        for d in parsed.get("defended_homogeneous_classes", []):
            extra.append(f"- {d.get('class_id')}: {d.get('rationale', '')[:140]}")
    if parsed.get("singletons_review"):
        force_merge = [
            s
            for s in parsed["singletons_review"]
            if s.get("verdict") == "force_merge_at_final"
        ]
        if force_merge:
            extra.append(
                f"**Singletons flagged force_merge_at_final:** "
                f"{[s['class_id'] for s in force_merge]}"
            )
    _append_to_watch_items(
        "consolidation", consolidation_idx, parsed.get("summary", ""), extra_lines=extra
    )
    # Persist surfaced_sub_channels emitted by consolidation
    source_label = f"consolidation_{consolidation_idx:03d}"
    n_subs = _append_surfaced_sub_channels(parsed, source_label)
    if n_subs:
        print(f"  persisted {n_subs} surfaced sub-channels to watchlist", flush=True)
    _append_step_log(
        step_type="consolidation",
        summary=parsed.get("summary", "")[:300],
        extra={
            "consolidation_idx": consolidation_idx,
            "merges": n_merges,
            "renames": n_renames,
            "axis_extensions": n_axis_ext,
            "splits_applied": splits_applied_count,
            "reassignments_applied": reassigns_applied,
            "split_dives_scheduled": len(parsed.get("split_dives_scheduled", [])),
            "defended_homogeneous": len(parsed.get("defended_homogeneous_classes", [])),
            "surfaced_sub_channels": n_subs,
        },
    )

    print(f"\n=== Consolidation {consolidation_idx:03d} complete ===", flush=True)
    print(f"  {applied_summary}", flush=True)
    print(
        f"  scheduled deep-dives: {len(parsed.get('split_dives_scheduled', []))}",
        flush=True,
    )
    print(
        f"  defended-homogeneous classes: "
        f"{len(parsed.get('defended_homogeneous_classes', []))}",
        flush=True,
    )
    print(f"  summary: {parsed.get('summary', '')[:300]}", flush=True)


def run_final_audit():
    catalog = _load_active_catalog()
    hc_counts, mc_counts = _compute_class_counts()
    total_assigned = sum(hc_counts.values())
    sentinel = uuid.uuid4().hex[:12]
    prompt = make_final_audit_prompt(
        catalog, hc_counts, mc_counts, total_assigned, sentinel
    )
    print(
        f"final audit prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)",
        flush=True,
    )
    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        "final_audit",
        PARTIAL_FP,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(f"FINAL AUDIT FAILED ({err})", flush=True)
        return
    try:
        parsed, method = _robust_json_parse(json_part, '{"singleton_decisions":')
        if method != "direct":
            print(f"  RECOVERED via {method}", flush=True)
    except json.JSONDecodeError as e:
        print(f"JSON parse unrecoverable: {e}", flush=True)
        return

    M.atomic_write_json(
        FINAL_AUDIT_FP,
        {
            "duration_sec": dur,
            "total_paths_assigned": total_assigned,
            "n_hc_in_audit": len(catalog["harm_classes"]),
            "n_mc_in_audit": len(catalog["mechanism_classes"]),
            "raw_output": parsed,
        },
    )
    # Auto-append final audit findings to watch_items
    extra = []
    if parsed.get("matrix_concerns"):
        extra.append(f"**Matrix concerns:** {parsed['matrix_concerns']}")
    if parsed.get("transferability_examples"):
        extra.append(
            f"**Transferability examples:** {len(parsed['transferability_examples'])} MCs touching multiple HCs (deliverable 2)"
        )
    if parsed.get("gap_candidates"):
        extra.append(
            f"**Gap candidates:** {len(parsed['gap_candidates'])} under-served HCs (deliverable 3)"
        )
    _append_to_watch_items(
        "final_audit", 1, parsed.get("summary", ""), extra_lines=extra
    )
    _append_step_log(
        step_type="final_audit",
        summary=parsed.get("summary", "")[:300],
        extra={
            "singleton_decisions": len(parsed.get("singleton_decisions", [])),
            "class_tightenings": len(parsed.get("class_tightenings", [])),
            "transferability_examples": len(parsed.get("transferability_examples", [])),
            "gap_candidates": len(parsed.get("gap_candidates", [])),
        },
    )
    print("\n=== Final audit complete ===", flush=True)
    print(f"  wrote {FINAL_AUDIT_FP.name}", flush=True)
    print(
        f"  singleton decisions: {len(parsed.get('singleton_decisions', []))}",
        flush=True,
    )
    print(
        f"  class tightenings: {len(parsed.get('class_tightenings', []))}", flush=True
    )
    print("  appended summary to watch_items", flush=True)
    print(f"  summary: {parsed.get('summary', '')[:300]}", flush=True)


# ============================================================
# Axes-only review pass
# ============================================================
def make_axes_review_prompt(
    catalog, sampled_assignments, node_attrs, paths_idx, sentinel
):
    """Show Opus a sample of (path + 6 axis values) and ask: which axis values
    look wrong/inconsistent? Targeted at axis robustness rather than HC/MC."""
    axes_str = "\n".join(
        f"  {ax['axis_name']} ({ax['axis_kind']}): {ax['values']}"
        for ax in catalog.get("axes", [])
    )
    watch_str = _load_watch_items() or "(no watch items recorded yet)"
    lines = []
    for a in sampled_assignments:
        p = paths_idx.get(a["path_id"])
        if not p:
            continue
        lines.append(M.fmt_path(p, node_attrs))
        ax_str = ", ".join(f"{k}={v}" for k, v in (a.get("axes") or {}).items())
        lines.append(f"  -> CURRENT_AXES: {ax_str}")
        lines.append(
            f"     HC={a.get('harm_class_id')} MC={a.get('mechanism_class_id')}\n"
        )
    body = "\n".join(lines)

    return f"""You are auditing axis assignments for the doublet catalog.

{M.PAPER_DELIVERABLE_CONTEXT}

============================================================
ACTIVE WATCH-ITEMS (prior reviews may have flagged axis vocab issues)
============================================================

{watch_str}

============================================================
AXES (controlled vocabularies)
============================================================

{axes_str}

============================================================
SAMPLE (path + currently-assigned axis values)
============================================================

{body}

============================================================
YOUR TASK
============================================================

For each path, evaluate whether its 6 axis values are correct given the path's
risk node + intervention node + body. Focus on:
  - severity / harm_target / emergence_stage (risk-side) — do they actually
    reflect the risk node's framing? Common error: defaulting to
    `catastrophic-existential / human-survival` when the path is moderate-scope.
  - lifecycle_stage / modality / methodology (intervention-side) — do they
    reflect the intervention node? Common error: `modality=general` overuse.
  - cross-axis consistency: `severity=catastrophic-existential` should rarely
    co-occur with `harm_target=economic`; `emergence_stage=training-time` should
    rarely co-occur with `lifecycle_stage=deployment-runtime`; flag mismatches.

OUTPUT — for each path where you'd change ≥1 axis value:
  - path_id
  - changes: [{{"axis": "modality", "from": "general", "to": "training-pipeline-infra (OTHER)", "rationale": "..."}}, ...]

Also output:
  - vocab_observations: axis values that are overused/underused/missing
  - cross_axis_correlations: any consistency rules that should be enforced

============================================================
OUTPUT FORMAT (STRICT)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.

Schema:

{{
  "axis_corrections": [
    {{"path_id": "path_NNNNN_dedup",
      "changes": [{{"axis": "modality", "from": "general", "to": "LLM", "rationale": "..."}}]}}
  ],
  "vocab_observations": [{{"axis": "modality", "observation": "general overused (N=X)", "recommendation": "..."}}],
  "cross_axis_correlations": [{{"rule": "severity=catastrophic-existential => harm_target in {{human-survival, institutional-governance}}", "violations": N}}],
  "summary": "<3-5 sentences>"
}}END_SENTINEL_{sentinel}

Produce the audit now."""


def run_axes_review(sample_n=50):
    """Sample N assignments uniformly and ask Opus to audit axis values."""
    catalog = _load_active_catalog()
    if not ASSIGNMENTS_FP.exists():
        print("No assignments to audit.", flush=True)
        return
    rows = [
        json.loads(line)
        for line in ASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < sample_n:
        sample_n = len(rows)
    rng = random.Random(ROUTING_RNG_SEED + 1)
    sample = rng.sample(rows, sample_n)
    paths = _load_deduped_paths()
    paths_idx = {p["path_id"]: p for p in paths}
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)

    state = _load_state()
    axes_idx = state.get("total_axes_reviews_run", 0) + 1
    sentinel = uuid.uuid4().hex[:12]
    prompt = make_axes_review_prompt(catalog, sample, na, paths_idx, sentinel)
    print(
        f"axes_review #{axes_idx:03d}: sampling {sample_n} assignments, prompt {len(prompt)} chars",
        flush=True,
    )
    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        f"axes_review_{axes_idx:03d}",
        PARTIAL_FP,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(f"axes_review failed: {err}", flush=True)
        return
    try:
        parsed, method = _robust_json_parse(json_part, '{"axis_corrections":')
        if method != "direct":
            print(f"  RECOVERED via {method}", flush=True)
    except json.JSONDecodeError as e:
        print(f"JSON parse unrecoverable: {e}", flush=True)
        return

    out_dir = STEP1 / "phase2_routing_axes_reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"axes_review_{axes_idx:03d}.json"
    M.atomic_write_json(
        out_fp,
        {
            "axes_review_idx": axes_idx,
            "duration_sec": dur,
            "sample_n": sample_n,
            "n_corrections": len(parsed.get("axis_corrections", [])),
            "raw_output": parsed,
        },
    )
    print(
        f"wrote {out_fp.name}: {len(parsed.get('axis_corrections', []))} corrections, "
        f"{len(parsed.get('vocab_observations', []))} vocab observations",
        flush=True,
    )
    state["total_axes_reviews_run"] = axes_idx
    _save_state(state)
    # Auto-append axes review to watch_items
    extra = []
    if parsed.get("vocab_observations"):
        extra.append("**Vocab observations:**")
        for o in parsed.get("vocab_observations", []):
            extra.append(
                f"- {o.get('axis')}: {o.get('observation', '')[:100]} -> {o.get('recommendation', '')[:100]}"
            )
    _append_to_watch_items(
        "axes_review", axes_idx, parsed.get("summary", ""), extra_lines=extra
    )
    _append_step_log(
        step_type="axes_review",
        summary=parsed.get("summary", "")[:300],
        extra={
            "axes_review_idx": axes_idx,
            "sample_n": sample_n,
            "axis_corrections": len(parsed.get("axis_corrections", [])),
            "vocab_observations": len(parsed.get("vocab_observations", [])),
        },
    )
    print("  appended summary to watch_items", flush=True)


# ============================================================
# Misfit review (Opus-adjudicated per-consolidation pass over heuristic candidates)
# ============================================================
def make_misfit_review_prompt(
    catalog, candidates, hc_counts, mc_counts, node_attrs, paths_idx, sentinel
):
    """Show Opus the top-K candidate misfits (heuristic + low-fit) alongside
    full catalog, ask which need reassignment vs which holds + propose target."""
    catalog_str = _fmt_catalog_for_routing(catalog, hc_counts, mc_counts)
    watch_str = _load_watch_items() or "(no watch items recorded yet)"
    # WATCHLIST: cumulative sub-channels named by prior reviews but lacking
    # >=MIN_GROUP_SIZE quorum. Plus the LAST misfit_review's surfaced
    # sub-channels in a separate slot so Opus weights recency explicitly.
    sub_channels_all_str = _load_surfaced_sub_channels_for_prompt()
    last_misfit_idx = _load_state().get("total_misfit_reviews_run", 0)
    sub_channels_last_str = (
        _load_surfaced_sub_channels_for_prompt(
            only_last_source=f"misfit_review_{last_misfit_idx:03d}"
        )
        if last_misfit_idx
        else "(no prior misfit review)"
    )
    lines = []
    for c in candidates:
        p = paths_idx.get(c["path_id"])
        if not p:
            continue
        lines.append(M.fmt_path(p, node_attrs))
        lines.append(
            f"  -> CURRENT: HC={c.get('harm_class_id')} "
            f"MC={c.get('mechanism_class_id')} "
            f"fit_score={c.get('fit_score')} "
            f"axes={c.get('axes')}"
        )
        if c.get("fit_note"):
            lines.append(f"     fit_note: {c['fit_note']}")
        if c.get("flag_reason"):
            lines.append(f"     flag_reason: {c['flag_reason']}")
        lines.append("")
    body = "\n".join(lines)
    return f"""You are doing a MISFIT REVIEW PASS over the doublet catalog.

{M.PAPER_DELIVERABLE_CONTEXT}

============================================================
ACTIVE WATCH-ITEMS (prior review outputs — recent batches, sweeps, consolidations, misfit_reviews)

Use these to: (i) avoid contradicting prior adjudication unless evidence warrants;
(ii) refer to known catalog gaps when judging PROPOSE_NEW; (iii) align with the
defended-as-homogeneous decisions from oversized-class deep-dives.
============================================================

{watch_str}

============================================================
SURFACED SUB-CHANNELS WATCHLIST (cumulative across all prior reviews)

These sub-channels were named by prior reviews as gaps but did not yet reach
the >=3-peer quorum to be proposed as new HC/MC. When a candidate in this
review matches one of these gaps, INCLUDE the matching path_id in your
PROPOSE_NEW.additional_path_candidates so the peer count accumulates. If a
sub-channel's total peer count (existing watchlist + this review's new peers)
reaches >=3, you MAY now PROPOSE_NEW with the full cumulative peer set cited.
============================================================

{sub_channels_all_str}

============================================================
LAST MISFIT_REVIEW SURFACED SUB-CHANNELS (most recent only — for recency weighting)
============================================================

{sub_channels_last_str}

============================================================
CURRENT CATALOG
============================================================

{catalog_str}

============================================================
CANDIDATE MISFITS — paths surfaced by 5 sources (check `flag_reason`):
  (a) Opus self-flagged fit_score <= 3 (low-fit)
  (b) Heuristic keyword-overlap audit (low overlap with assigned class)
  (c) Manual reassign_pending tag (from prior audit)
  (d) UNASSIGNED paths — currently lack HC and/or MC, awaiting peers
  (e) OVERSIZE_SAMPLE — random sample from classes with n >= 25; review
      whether they actually fit the catch-all or belong to a finer sub-class
============================================================

{body}

============================================================
YOUR TASK
============================================================

For each path above, decide ONE of:

  - HOLD — current (HC, MC) is correct despite the flag. Explain WHY the
    fit IS appropriate. For oversize_sample candidates, HOLD means the path
    legitimately fits the catch-all because that class is genuinely one
    coherent mechanism family (cite member-pair similarity).

  - REASSIGN_HC — propose a better EXISTING HC. Cite the risk-side evidence.
    Use for low-fit and unassigned-HC paths that fit an existing class.

  - REASSIGN_MC — propose a better EXISTING MC. Cite the mech-side evidence.

  - PROPOSE_NEW_HC — propose a new HC. **REQUIRED**: cite >=2 OTHER
    `path_id`s in `additional_path_candidates` (giving >=3 members with the
    focus path). Singletons FORBIDDEN — code drops the proposal otherwise.
    Use for: (i) unassigned-HC clusters where >=3 unassigned share a sub-channel,
    (ii) oversize_sample paths where you can identify >=3 paths needing a
    finer sub-class carved from the catch-all.

  - PROPOSE_NEW_MC — same for mechanism class. Requires >=3 members.

  - UNASSIGN — path genuinely doesn't fit any existing or proposable class.
    Use sparingly — only when truly out-of-scope (non-AI-safety, niche topic
    with no peers expected). Pre-existing unassigned paths can be HOLD-as-
    unassigned with the same intent.

NO-HETEROGENEOUS-CATCH-ALL POLICY: For oversize_sample candidates, if the
class has ≥2 distinct sub-mechanism families with ≥3 members each, those
sub-channels MUST be split out via PROPOSE_NEW. Defense as "broad-but-
coherent" is FORBIDDEN — argue homogeneity with concrete member-pair
similarity if claiming HOLD on an oversize_sample.

Use risk/intervention DECOUPLING: re-evaluate the risk-side (risk node + 1-2
body) and mechanism-side (last 1-2 body + intervention) independently. Don't
let the current assignment anchor your judgment.

============================================================
OUTPUT FORMAT (STRICT)
============================================================

- ONLY a JSON object. No preamble. No markdown.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}`.

Schema:

{{
  "decisions": [
    {{"path_id": "path_NNNNN_dedup",
      "verdict": "HOLD" | "REASSIGN_HC" | "REASSIGN_MC" | "PROPOSE_NEW_HC" | "PROPOSE_NEW_MC" | "UNASSIGN",
      "new_hc_id": "<HC### if REASSIGN_HC>",
      "new_mc_id": "<MC### if REASSIGN_MC>",
      "new_hc_proposal": {{"class_name": "...", "class_description": "...", "is_capability_gap": false, "additional_path_candidates": ["path_NNNNN_dedup"]}},
      "new_mc_proposal": {{"class_name": "...", "class_description": "...", "additional_path_candidates": ["path_NNNNN_dedup"]}},
      "rationale": "<1-2 sentences>"}}
  ],
  "surfaced_sub_channels": [
    {{"name": "<short kebab-or-prose name for the gap, e.g. 'Bayesian-optimization-for-hyperparameter-search'>",
      "side": "HC" | "MC",
      "rationale": "<1-2 sentences: what mechanism family or risk class this represents>",
      "named_peer_path_ids": ["path_NNNNN_dedup", "..."],
      "n_peers_in_corpus_so_far": <int>,
      "min_required": 3,
      "confirmed_out_of_scope": <true if this sub-channel is non-AI-safety and should remain UNASSIGNED even at quorum; false otherwise>}}
  ],
  "summary": "<3-5 sentences: how many HOLD/REASSIGN/etc, any patterns observed>"
}}END_SENTINEL_{sentinel}

NOTE on `surfaced_sub_channels`: if you spot a coherent gap in the catalog that
this review's candidates partially populate but lacks >=3 named peers in this
review's input, emit it here so future reviews can accumulate peers across
cycles. If the watchlist above shows this sub-channel was already named in a
prior review, ADD any new in-this-review peer paths to `named_peer_path_ids`
(the system will dedupe across sources). If watchlist count + this review's
new peers reaches >=3, you SHOULD instead emit a PROPOSE_NEW_HC/PROPOSE_NEW_MC
decision with the full cumulative peer list.

IMPORTANT — `confirmed_out_of_scope` flag: set to true for sub-channels that
are non-AI-safety (e.g. AI applied to non-AI cybersecurity, non-AI
cryptography, classical control engineering, conservation AI, human cognitive
training, corporate finance, generic ML productivity tooling, non-AI
existential cause areas). For OOS sub-channels we DO NOT propose new classes
even at quorum — matching paths stay UNASSIGNED. The flag prevents
re-surfacing in future reviews.

Produce the review now."""


def run_misfit_review(
    top_k_per_class=5,
    max_candidates=200,
    oversize_sample_n=8,
    scope="all",
    exclude_previously_reviewed=True,
):
    """Per-consolidation Opus pass adjudicating misfit candidates.

    Candidate set (expanded 2026-05-18):
      (a) Low-fit self-flagged (fit_score <= 3)
      (b) Heuristic keyword-overlap misfits
      (c) Manual reassign_pending tag
      (d) Unassigned paths (harm_class_id=None OR mechanism_class_id=None) —
          Opus decides: HOLD-unassigned, REASSIGN-to-existing-class, or
          PROPOSE_NEW with peer cluster
      (e) Random sample of N=`oversize_sample_n` per oversized class
          (n >= OVERSIZE_ALARM) — Opus checks if members actually belong to
          the catch-all or to a finer sub-class via PROPOSE_NEW with peer paths

    `scope` selector (added 2026-05-19 per R1/R2 split in §19.16):
      "all"          — include (a)+(b)+(c)+(d)+(e) (default; legacy behavior)
      "self_flagged" — only (a)+(b)+(c) — R1 self-flagged misfit pass
      "unassigned"   — only (d) — R2 unassigned-pool audit
      "oversize"     — only (e) — legacy oversize sampling (superseded by
                        final_misfit_sweep deep-dives)

    `exclude_previously_reviewed` (added 2026-05-20 — no-rework filter):
      When True, exclude paths previously HOLDed in any misfit_review_NNN.json
      UNLESS the path has been re-flagged since (appears in a batch flag's
      reassign_candidates OR appears in path_reassignments log AFTER its last
      HOLD verdict). This prevents R1' / R2' from re-reviewing paths whose
      classification was already validated in a prior cycle.

      Default True. Set False only for re-audit purposes (e.g. after a
      catalog overhaul that invalidates prior verdicts).

    All changes persisted via path_reassignments log so they survive future
    _rebuild_routing_assignments_jsonl calls.
    """
    include_self_flagged = scope in ("all", "self_flagged")
    include_unassigned = scope in ("all", "unassigned")
    include_oversize = scope in ("all", "oversize")

    # Build "previously HOLDed in misfit_review" set + their last-HOLD step idx
    prev_held_by_pid = {}  # pid -> last misfit_review_idx that HOLDed it
    mr_dir = STEP1 / "phase2_routing_misfit_reviews"
    if exclude_previously_reviewed and mr_dir.exists():
        for fp in sorted(mr_dir.glob("misfit_review_*.json")):
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            mr_idx = d.get("misfit_review_idx") or 0
            for dec in d.get("raw_output", {}).get("decisions", []):
                if dec.get("verdict") in ("HOLD",):
                    pid = dec.get("path_id")
                    if pid:
                        prev_held_by_pid[pid] = max(
                            prev_held_by_pid.get(pid, 0), mr_idx
                        )
    # Subtract: paths whose reassignment log shows movement AFTER their last HOLD
    # (e.g. consolidation reassigned them, so they're effectively re-flagged)
    if exclude_previously_reviewed and PATH_REASSIGNMENTS_FP.exists():
        for line in PATH_REASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = e.get("path_id")
            if pid and pid in prev_held_by_pid:
                # Approximation: any post-HOLD reassign log entry invalidates the
                # HOLD. (Fine-grained: parse source label for misfit_review_NNN
                # idx and compare; for now, any movement means re-eligible.)
                src = e.get("source", "")
                # If source is from a misfit_review with idx > last HOLD, re-eligible
                if "misfit_review_" in src:
                    try:
                        src_idx = int(src.split("misfit_review_")[1].split("_")[0])
                        if src_idx > prev_held_by_pid[pid]:
                            del prev_held_by_pid[pid]
                    except (ValueError, IndexError):
                        pass
                # Movements from consolidation/sweep — assume newer than last HOLD
                # since these are append-only and run after HOLDs in chronological order
                elif "consolidation_" in src or "final_sweep_" in src:
                    del prev_held_by_pid[pid]
    n_excluded_by_prev_hold = len(prev_held_by_pid)
    catalog = _load_active_catalog()
    if not ASSIGNMENTS_FP.exists():
        print("No assignments. Nothing to review.", flush=True)
        return
    rows = [
        json.loads(line)
        for line in ASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hc_counts, mc_counts = _compute_class_counts()

    cands = {}
    n_self_flagged = 0
    n_unassigned_added = 0
    n_oversize_added = 0

    if include_self_flagged:
        # (a) low-fit self-flagged + (c) reassign_pending
        for r in rows:
            pid = r["path_id"]
            if pid in prev_held_by_pid:
                continue  # no-rework filter — previously HOLDed and not re-flagged
            if r.get("fit_score") is not None and r["fit_score"] <= 3:
                cands[pid] = dict(
                    r, flag_reason=f"low_fit_self_flag (fit={r['fit_score']})"
                )
                n_self_flagged += 1
            if r.get("reassign_pending"):
                cands[pid] = dict(r, flag_reason="manual reassign_pending tag")
                n_self_flagged += 1
        # (b) heuristic misfits
        heur_fp = STEP1 / "phase2_routing_heuristic_misfits.json"
        if heur_fp.exists():
            try:
                hm = json.loads(heur_fp.read_text(encoding="utf-8"))
                for r in hm.get("top_hc_misfits", []) + hm.get("top_mc_misfits", []):
                    pid = r["path_id"]
                    if pid in prev_held_by_pid:
                        continue  # no-rework filter
                    if pid not in cands:
                        full = next((x for x in rows if x["path_id"] == pid), None)
                        if full:
                            cands[pid] = dict(
                                full,
                                flag_reason=f"heuristic low-overlap (HC={r.get('risk_overlap', '?')} MC={r.get('mech_overlap', '?')})",
                            )
                            n_self_flagged += 1
            except Exception as e:
                print(f"heuristic misfit read error: {e}", flush=True)

    if include_unassigned:
        # (d) Unassigned paths
        for r in rows:
            pid = r["path_id"]
            if pid in prev_held_by_pid:
                continue  # no-rework filter — already confirmed unassigned or held
            is_unassigned_hc = (
                r.get("harm_class_id") is None
                or r.get("harm_class_status") == "unassigned"
            )
            is_unassigned_mc = (
                r.get("mechanism_class_id") is None
                or r.get("mechanism_class_status") == "unassigned"
            )
            if (is_unassigned_hc or is_unassigned_mc) and pid not in cands:
                sides = []
                if is_unassigned_hc:
                    sides.append("HC")
                if is_unassigned_mc:
                    sides.append("MC")
                cands[r["path_id"]] = dict(
                    r,
                    flag_reason=f"unassigned ({','.join(sides)}): {(r.get('fit_note') or '')[:120]}",
                )
                n_unassigned_added += 1

    if include_oversize:
        # (e) Random sample of N per oversized class — surfaces actual catch-all members
        rng_sample = random.Random(ROUTING_RNG_SEED + 2)
        oversized_hc = sorted([h for h, n in hc_counts.items() if n >= OVERSIZE_ALARM])
        oversized_mc = sorted([m for m, n in mc_counts.items() if n >= OVERSIZE_ALARM])
        for cls in oversized_hc:
            members = [r for r in rows if r.get("harm_class_id") == cls]
            sample = rng_sample.sample(members, min(oversize_sample_n, len(members)))
            for r in sample:
                if r["path_id"] not in cands:
                    cands[r["path_id"]] = dict(
                        r,
                        flag_reason=f"oversize_sample HC={cls} (n={hc_counts[cls]}); review if path "
                        f"actually fits this class or a finer sub-channel",
                    )
                    n_oversize_added += 1
        for cls in oversized_mc:
            members = [r for r in rows if r.get("mechanism_class_id") == cls]
            sample = rng_sample.sample(members, min(oversize_sample_n, len(members)))
            for r in sample:
                if r["path_id"] not in cands:
                    cands[r["path_id"]] = dict(
                        r,
                        flag_reason=f"oversize_sample MC={cls} (n={mc_counts[cls]}); review if path "
                        f"actually fits this class or a finer sub-channel",
                    )
                    n_oversize_added += 1

    print(
        f"misfit_review candidates (scope={scope}): {len(cands)} total "
        f"(+{n_self_flagged} self_flagged, +{n_unassigned_added} unassigned, "
        f"+{n_oversize_added} oversize-sample); "
        f"excluded {n_excluded_by_prev_hold} previously-HOLDed (no-rework filter)",
        flush=True,
    )

    if not cands:
        print("No misfit candidates. Skipping review.", flush=True)
        return

    if len(cands) > max_candidates:

        def priority(r):
            fr = r.get("flag_reason", "")
            if "unassigned" in fr:
                return 0
            if "low_fit" in fr or "reassign_pending" in fr:
                return 1
            if "heuristic" in fr:
                return 2
            return 3  # oversize_sample

        scored = sorted(
            cands.values(), key=lambda r: (priority(r), r.get("fit_score") or 5)
        )
        cands = {r["path_id"]: r for r in scored[:max_candidates]}
        print(
            f"  capped to {len(cands)} (priority: unassigned > low-fit > heuristic > oversize-sample)",
            flush=True,
        )

    paths = _load_deduped_paths()
    paths_idx = {p["path_id"]: p for p in paths}
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    hc_counts, mc_counts = _compute_class_counts()
    state = _load_state()
    mr_idx = state.get("total_misfit_reviews_run", 0) + 1
    sentinel = uuid.uuid4().hex[:12]
    prompt = make_misfit_review_prompt(
        catalog, list(cands.values()), hc_counts, mc_counts, na, paths_idx, sentinel
    )
    print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)

    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        f"misfit_review_{mr_idx:03d}",
        PARTIAL_FP,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(f"misfit_review failed: {err}", flush=True)
        return
    try:
        parsed, method = _robust_json_parse(json_part, '{"decisions":')
        if method != "direct":
            print(f"  RECOVERED via {method}", flush=True)
    except json.JSONDecodeError as e:
        print(f"JSON parse unrecoverable: {e}", flush=True)
        return

    out_dir = STEP1 / "phase2_routing_misfit_reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"misfit_review_{mr_idx:03d}.json"
    M.atomic_write_json(
        out_fp,
        {
            "misfit_review_idx": mr_idx,
            "duration_sec": dur,
            "n_candidates": len(cands),
            "n_decisions": len(parsed.get("decisions", [])),
            "raw_output": parsed,
        },
    )

    # Apply decisions via path_reassignments log + catalog mutations for
    # PROPOSE_NEW. Changes survive future _rebuild_routing_assignments_jsonl
    # calls because the override log is read at rebuild time.
    verdict_counter = Counter()
    for d in parsed.get("decisions", []):
        verdict_counter[d.get("verdict", "?")] += 1
    print(f"  verdicts: {dict(verdict_counter)}", flush=True)

    applied = 0
    new_classes_created = []
    proposed_classes_dropped = []
    catalog_dirty = False

    for d in parsed.get("decisions", []):
        pid = d.get("path_id", "")
        v = d.get("verdict")
        source = f"misfit_review_{mr_idx:03d}"
        rat = d.get("rationale", "")
        if v == "REASSIGN_HC" and d.get("new_hc_id"):
            _append_path_reassignment(
                pid,
                source=f"{source}_REASSIGN_HC",
                new_hc=d["new_hc_id"],
                rationale=rat,
            )
            applied += 1
        elif v == "REASSIGN_MC" and d.get("new_mc_id"):
            _append_path_reassignment(
                pid,
                source=f"{source}_REASSIGN_MC",
                new_mc=d["new_mc_id"],
                rationale=rat,
            )
            applied += 1
        elif v == "UNASSIGN":
            _append_path_reassignment(
                pid,
                source=f"{source}_UNASSIGN",
                new_hc=_SENTINEL_UNASSIGN,
                new_mc=_SENTINEL_UNASSIGN,
                rationale=rat,
            )
            applied += 1
        elif v == "PROPOSE_NEW_HC":
            prop = d.get("new_hc_proposal") or {}
            members = [pid] + list(prop.get("additional_path_candidates") or [])
            # Dedup while preserving order
            seen = set()
            members = [m for m in members if not (m in seen or seen.add(m))]
            if len(members) < MIN_GROUP_SIZE or not prop.get("class_name"):
                proposed_classes_dropped.append(
                    ("HC", pid, len(members), prop.get("class_name", "")[:40])
                )
                continue
            new_id = _next_class_id(catalog, "HC")
            catalog["harm_classes"].append(
                {
                    "class_id": new_id,
                    "class_name": prop["class_name"],
                    "class_description": prop.get("class_description", ""),
                    "is_capability_gap": bool(prop.get("is_capability_gap", False)),
                    "source": f"misfit_review_{mr_idx:03d}_propose_new",
                }
            )
            catalog_dirty = True
            new_classes_created.append((new_id, prop["class_name"], len(members)))
            for m_pid in members:
                _append_path_reassignment(
                    m_pid,
                    source=f"{source}_PROPOSE_NEW_HC",
                    new_hc=new_id,
                    rationale=f"new HC: {prop.get('class_description', '')[:200]}",
                )
                applied += 1
        elif v == "PROPOSE_NEW_MC":
            prop = d.get("new_mc_proposal") or {}
            members = [pid] + list(prop.get("additional_path_candidates") or [])
            seen = set()
            members = [m for m in members if not (m in seen or seen.add(m))]
            if len(members) < MIN_GROUP_SIZE or not prop.get("class_name"):
                proposed_classes_dropped.append(
                    ("MC", pid, len(members), prop.get("class_name", "")[:40])
                )
                continue
            new_id = _next_class_id(catalog, "MC")
            catalog["mechanism_classes"].append(
                {
                    "class_id": new_id,
                    "class_name": prop["class_name"],
                    "class_description": prop.get("class_description", ""),
                    "source": f"misfit_review_{mr_idx:03d}_propose_new",
                }
            )
            catalog_dirty = True
            new_classes_created.append((new_id, prop["class_name"], len(members)))
            for m_pid in members:
                _append_path_reassignment(
                    m_pid,
                    source=f"{source}_PROPOSE_NEW_MC",
                    new_mc=new_id,
                    rationale=f"new MC: {prop.get('class_description', '')[:200]}",
                )
                applied += 1
        # HOLD: no action

    if catalog_dirty:
        _save_active_catalog(catalog, kind_suffix=f"misfit_review_{mr_idx:03d}")
    # Rebuild merged jsonl (applies the reassignment log)
    n_rows = _rebuild_routing_assignments_jsonl()
    print(
        f"  applied {applied} reassignments/unassigns/new-class-moves; "
        f"{len(new_classes_created)} new classes created; "
        f"{len(proposed_classes_dropped)} proposals dropped (<MIN_GROUP_SIZE); "
        f"merged jsonl now {n_rows} rows",
        flush=True,
    )
    for nc in new_classes_created:
        print(f"    new {nc[0]}: {nc[1]} ({nc[2]} members)", flush=True)
    for dp in proposed_classes_dropped:
        print(
            f"    DROPPED {dp[0]} proposal from {dp[1]}: only {dp[2]} members "
            f"('{dp[3]}')",
            flush=True,
        )
    state["total_misfit_reviews_run"] = mr_idx
    _save_state(state)
    # Auto-append pass summary + actions to watch_items
    summary = parsed.get("summary", "")
    extra = []
    if verdict_counter:
        extra.append(f"**Verdict counts:** {dict(verdict_counter)}")
    reassign_lines = []
    for d in parsed.get("decisions", []):
        v = d.get("verdict")
        if v == "REASSIGN_HC":
            reassign_lines.append(
                f"- {d['path_id']}: HC -> {d.get('new_hc_id')} ({d.get('rationale', '')[:140]})"
            )
        elif v == "REASSIGN_MC":
            reassign_lines.append(
                f"- {d['path_id']}: MC -> {d.get('new_mc_id')} ({d.get('rationale', '')[:140]})"
            )
    if reassign_lines:
        extra.append("**Reassignments applied:**")
        extra.extend(reassign_lines)
    _append_to_watch_items("misfit_review", mr_idx, summary, extra_lines=extra)
    # Persist surfaced_sub_channels to sidecar — picked up by future Opus calls
    source_label = f"misfit_review_{mr_idx:03d}"
    n_subs = _append_surfaced_sub_channels(parsed, source_label)
    if n_subs:
        print(f"  persisted {n_subs} surfaced sub-channels to watchlist", flush=True)
    # Step log entry — chronological audit trail
    _append_step_log(
        step_type="misfit_review",
        summary=summary or f"{applied} reassignments / {len(cands)} candidates",
        extra={
            "misfit_review_idx": mr_idx,
            "scope": scope,
            "candidates": len(cands),
            "applied": applied,
            "new_classes": locals().get("new_classes_created", 0),
            "surfaced_sub_channels": n_subs,
        },
    )
    print(
        f"  applied {applied} reassignments / unassignments to merged jsonl", flush=True
    )
    print("  appended summary to watch_items", flush=True)
    print(f"  wrote {out_fp.name}", flush=True)


# ============================================================
# Final misfit sweep — comprehensive per-class member walk at end-of-run
# ============================================================
def make_class_sweep_prompt(
    catalog,
    class_kind,
    class_entry,
    member_rows,
    all_classes_str,
    node_attrs,
    paths_idx,
    sentinel,
):
    """For one class, ask Opus to review every member and flag misfits.

    class_kind in {"HC", "MC"}.
    member_rows: full assignment rows for paths in this class.
    """
    watch_str = _load_watch_items() or "(no watch items recorded yet)"
    sub_channels_str = _load_surfaced_sub_channels_for_prompt()
    lines = []
    for r in member_rows:
        p = paths_idx.get(r["path_id"])
        if not p:
            continue
        lines.append(M.fmt_path(p, node_attrs))
        lines.append(
            f"  -> CURRENT: HC={r.get('harm_class_id')} "
            f"MC={r.get('mechanism_class_id')} "
            f"fit={r.get('fit_score')} conf={r.get('confidence')}"
        )
        if r.get("fit_note"):
            lines.append(f"     fit_note: {r['fit_note']}")
        lines.append("")
    members_body = "\n".join(lines)

    if class_kind == "HC":
        side = "harm-class"
        side_evidence = "risk node + first 1-2 body nodes"
    else:
        side = "mechanism-class"
        side_evidence = "last 1-2 body nodes + intervention node"

    return f"""You are doing the FINAL MISFIT SWEEP for one {side} in the doublet catalog.

{M.PAPER_DELIVERABLE_CONTEXT}

============================================================
ACTIVE WATCH-ITEMS (prior reviews — sweeps, consolidations, misfit_reviews)
============================================================

{watch_str}

============================================================
SURFACED SUB-CHANNELS WATCHLIST (gaps named by prior reviews; cumulative peers)

When reviewing members of this class, EXPLICITLY check whether watchlist
sub-channels that target THIS class (or its descendants) match any members.
If a watchlist sub-channel names members of this class as peer candidates
AND you confirm the sub-channel is coherent AND ≥3 members fit it,
EMIT a SPLIT_OUT decision for the focal member with the watchlist's peer
list (deduped against current members of this class).

If a watchlist sub-channel does NOT match any members of this class on
review, no action needed (it stays on the watchlist).
============================================================

{sub_channels_str}

============================================================
TARGET CLASS UNDER REVIEW
============================================================

class_id: {class_entry["class_id"]}
class_name: {class_entry["class_name"]}
class_description: {class_entry.get("class_description", "")}
is_capability_gap: {class_entry.get("is_capability_gap", False)}
n_current_members: {len(member_rows)}

============================================================
ALL OTHER CLASSES (potential reassignment targets)
============================================================

{all_classes_str}

============================================================
CURRENT MEMBERS OF THIS CLASS ({len(member_rows)} paths)
============================================================

{members_body}

============================================================
YOUR TASK
============================================================

For EVERY listed member, decide one of:
  - HOLD — path's {side_evidence} clearly fits the class definition above.
  - REASSIGN — path fits a different existing class better. Cite target class_id.
  - UNASSIGN — path doesn't fit this OR any other existing class well.
  - SPLIT_OUT — the path is genuinely distinct from peers AND >=2 other current
    members of THIS class would fit better in a new class with it. Cite the peer
    path_ids that would join.

Apply risk/intervention DECOUPLING — assess the {side_evidence} INDEPENDENTLY
from the other half. Many misfits arise because the intervention pattern
matched while the risk side did not (or vice versa).

Output ALL member decisions (no skipping). If you say HOLD, no rationale
needed; otherwise 1 clause rationale.

============================================================
OUTPUT FORMAT (STRICT)
============================================================

- ONLY a JSON object. No preamble. No markdown.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}`.

Schema:

{{
  "class_id": "{class_entry["class_id"]}",
  "decisions": [
    {{"path_id": "path_NNNNN_dedup",
      "verdict": "HOLD" | "REASSIGN" | "UNASSIGN" | "SPLIT_OUT",
      "target_class_id": "<if REASSIGN>",
      "split_proposal": {{"new_class_name": "...", "new_class_description": "...", "peer_path_ids": [...]}},
      "rationale": "<if not HOLD>"}}
  ],
  "surfaced_sub_channels": [
    {{"name": "<short kebab-or-prose name for the gap>",
      "side": "{class_kind}",
      "rationale": "<1-2 sentences>",
      "named_peer_path_ids": ["path_NNNNN_dedup", "..."],
      "n_peers_in_corpus_so_far": <int>,
      "min_required": 3,
      "confirmed_out_of_scope": <true if this sub-channel is non-AI-safety and should remain UNASSIGNED even at quorum; false otherwise>}}
  ],
  "summary": "<2-3 sentences: how many HOLD vs other; any patterns>"
}}END_SENTINEL_{sentinel}

NOTE on `surfaced_sub_channels`: if member review surfaces a coherent
sub-family with <3 named peers (so SPLIT_OUT is not warranted yet), emit
it here so future reviews can accumulate peers. If 1-2 in-class members
match a watchlist sub-channel listed above, ADD them to that sub-channel's
peer list (system dedupes across sources). When cumulative peers reach
>=3, emit SPLIT_OUT in `decisions` instead with the full peer list.

IMPORTANT — `confirmed_out_of_scope` flag: set to true for sub-channels
that are non-AI-safety (non-AI cybersecurity, non-AI cryptography,
classical control engineering, conservation AI, human cognitive training,
corporate finance, generic ML productivity tooling, non-AI x-risk causes).
OOS sub-channels stay UNASSIGNED — do NOT propose new classes for them
even at quorum. The flag prevents re-surfacing in future reviews.

Produce the sweep now."""


def run_final_misfit_sweep(chunk_size=100, max_classes=None, class_ids=None):
    """Comprehensive per-class member walk. For each HC and MC in scope,
    chunk the member list into batches of `chunk_size` and ask Opus to
    HOLD/REASSIGN/UNASSIGN/SPLIT_OUT each member.

    APPLIES SPLIT_OUT decisions IN-RUN (not deferred): groups co-cited
    SPLIT_OUT proposals by new_class_name, allocates new class ID for each
    group with >=MIN_GROUP_SIZE members, adds to catalog, persists path
    reassignments via the override log.

    Args:
      chunk_size: members per Opus chunk (default 40)
      max_classes: optional cap on number of classes processed
      class_ids: optional list/set of class IDs to filter to (e.g.,
                  ["HC002","MC015"]) — limits scope to specific classes,
                  ideal for mid-run "split deep-dive" use case

    Class A. Cost: roughly (n_chunks * ~25-30k tokens). At 9 chunks for
    6 deferred-class scope: ~35-40pp session.
    """
    catalog = _load_active_catalog()
    if not ASSIGNMENTS_FP.exists():
        print("No assignments. Skipping.", flush=True)
        return
    rows = [
        json.loads(line)
        for line in ASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    paths = _load_deduped_paths()
    paths_idx = {p["path_id"]: p for p in paths}
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)

    # Render brief "all other classes" listing for context
    hc_listing = "\n".join(
        f"  {h['class_id']}: {h['class_name']} — "
        f"{(h.get('class_description') or '')[:120]}"
        for h in catalog["harm_classes"]
    )
    mc_listing = "\n".join(
        f"  {m['class_id']}: {m['class_name']} — "
        f"{(m.get('class_description') or '')[:120]}"
        for m in catalog["mechanism_classes"]
    )
    hc_other_str = "HARM CLASSES:\n" + hc_listing
    mc_other_str = "MECHANISM CLASSES:\n" + mc_listing

    out_dir = STEP1 / "phase2_routing_final_misfit_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    sweep_idx = state.get("total_final_sweeps_run", 0) + 1
    all_decisions = []
    n_class_calls = 0

    classes_to_sweep = []
    filter_set = set(class_ids) if class_ids else None
    for h in catalog["harm_classes"]:
        if filter_set is None or h["class_id"] in filter_set:
            classes_to_sweep.append(("HC", h))
    for m in catalog["mechanism_classes"]:
        if filter_set is None or m["class_id"] in filter_set:
            classes_to_sweep.append(("MC", m))
    if max_classes:
        classes_to_sweep = classes_to_sweep[:max_classes]
    if filter_set is not None:
        found = {c["class_id"] for _, c in classes_to_sweep}
        missing = filter_set - found
        if missing:
            print(
                f"WARNING: requested class_ids not found in catalog: {sorted(missing)}",
                flush=True,
            )
    print(
        f"sweep #{sweep_idx} scope: {len(classes_to_sweep)} classes "
        f"{[c['class_id'] for _, c in classes_to_sweep]}",
        flush=True,
    )

    for class_kind, class_entry in classes_to_sweep:
        cid = class_entry["class_id"]
        if class_kind == "HC":
            members = [r for r in rows if r.get("harm_class_id") == cid]
            other_str = mc_other_str  # context for cross-side awareness
        else:
            members = [r for r in rows if r.get("mechanism_class_id") == cid]
            other_str = hc_other_str
        if not members:
            continue
        # Chunk if large
        n_chunks = (len(members) + chunk_size - 1) // chunk_size
        for ci in range(n_chunks):
            chunk = members[ci * chunk_size : (ci + 1) * chunk_size]
            sentinel = uuid.uuid4().hex[:12]
            prompt = make_class_sweep_prompt(
                catalog,
                class_kind,
                class_entry,
                chunk,
                other_str,
                na,
                paths_idx,
                sentinel,
            )
            print(
                f"sweep #{sweep_idx} {class_kind} {cid} chunk {ci + 1}/{n_chunks} "
                f"({len(chunk)} paths, {len(prompt)} chars) ...",
                flush=True,
            )
            json_part, dur, _, err = M.streaming_call_with_validation(
                prompt,
                sentinel,
                f"sweep_{sweep_idx:03d}_{cid}_c{ci:02d}",
                PARTIAL_FP,
                model="claude-opus-4-7",
            )
            if err or not json_part:
                print(f"  FAILED: {err}", flush=True)
                continue
            try:
                parsed, method = _robust_json_parse(json_part, '{"class_id":')
                if method != "direct":
                    print(f"  RECOVERED via {method}", flush=True)
            except json.JSONDecodeError as e:
                print(f"  JSON parse unrecoverable: {e}", flush=True)
                continue
            # Save chunk output
            out_fp = (
                out_dir / f"sweep_{sweep_idx:03d}_{class_kind}_{cid}_c{ci:02d}.json"
            )
            M.atomic_write_json(
                out_fp,
                {
                    "sweep_idx": sweep_idx,
                    "class_kind": class_kind,
                    "class_id": cid,
                    "chunk_idx": ci,
                    "n_chunks": n_chunks,
                    "n_in_chunk": len(chunk),
                    "duration_sec": dur,
                    "raw_output": parsed,
                },
            )
            for d in parsed.get("decisions", []):
                d["_class_kind"] = class_kind
                d["_class_id"] = cid
                all_decisions.append(d)
            # Persist any surfaced sub-channels Opus emitted for this class
            n_subs = _append_surfaced_sub_channels(
                parsed, source_label=f"final_sweep_{sweep_idx:03d}_{cid}"
            )
            if n_subs:
                print(
                    f"  persisted {n_subs} surfaced sub-channels to watchlist",
                    flush=True,
                )
            n_class_calls += 1

    # Apply decisions via path_reassignments log (durable across rebuilds).
    # SPLIT_OUT decisions are GROUPED by new_class_name within class_kind so
    # multiple co-cited splits merge into one new class.
    verdict_counter = Counter()
    applied = 0
    # First pass: process HOLD/REASSIGN/UNASSIGN
    for d in all_decisions:
        verdict_counter[d.get("verdict", "?")] += 1
        pid = d.get("path_id")
        if not pid:
            continue
        v = d.get("verdict")
        kind = d.get("_class_kind")
        if v == "HOLD":
            continue
        elif v == "REASSIGN" and d.get("target_class_id"):
            tgt = d["target_class_id"]
            kw = {"new_hc": tgt} if kind == "HC" else {"new_mc": tgt}
            _append_path_reassignment(
                pid,
                source=f"final_sweep_{sweep_idx:03d}_REASSIGN",
                rationale=d.get("rationale", ""),
                **kw,
            )
            applied += 1
        elif v == "UNASSIGN":
            kw = (
                {"new_hc": _SENTINEL_UNASSIGN}
                if kind == "HC"
                else {"new_mc": _SENTINEL_UNASSIGN}
            )
            _append_path_reassignment(
                pid,
                source=f"final_sweep_{sweep_idx:03d}_UNASSIGN",
                rationale=d.get("rationale", ""),
                **kw,
            )
            applied += 1
        # SPLIT_OUT handled in second pass below (needs grouping by class_name)

    # Second pass: SPLIT_OUT — group by (kind, NORMALIZED new_class_name) and
    # apply each group as a new class if it meets MIN_GROUP_SIZE.
    # The normalization (vs prior `.lower()`-only) was added 2026-05-20 after
    # R4 sweep #13 produced HC025+HC026 as cross-chunk duplicates that differed
    # only by a hyphen in 'speech perception' vs 'speech-perception'.
    split_groups = {}  # (kind, normalized_name) -> {"orig_class_id": ..., "name": ..., "desc": ..., "members": set()}
    for d in all_decisions:
        if d.get("verdict") != "SPLIT_OUT":
            continue
        pid = d.get("path_id")
        kind = d.get("_class_kind")
        from_cls = d.get("_class_id")
        sp = d.get("split_proposal") or {}
        new_name = (sp.get("new_class_name") or "").strip()
        new_desc = (sp.get("new_class_description") or "").strip()
        peers = sp.get("peer_path_ids") or []
        if not new_name or not pid:
            continue
        key = (kind, _normalize_class_name(new_name))
        if key not in split_groups:
            split_groups[key] = {
                "orig_class_id": from_cls,
                "name": new_name,
                "desc": new_desc,
                "members": set(),
                "is_capability_gap": sp.get("is_capability_gap", False),
            }
        split_groups[key]["members"].add(pid)
        for ppid in peers:
            if isinstance(ppid, str):
                split_groups[key]["members"].add(ppid)
        # Update description if a later proposal has a longer/non-empty one
        if new_desc and len(new_desc) > len(split_groups[key]["desc"]):
            split_groups[key]["desc"] = new_desc

    # Build NORMALIZED-name lookup against EXISTING catalog so a SPLIT_OUT
    # proposal whose name fuzzy-matches an existing class routes members into
    # that existing class instead of creating a duplicate.
    existing_hc_by_norm = {
        _normalize_class_name(h["class_name"]): h for h in catalog["harm_classes"]
    }
    existing_mc_by_norm = {
        _normalize_class_name(m["class_name"]): m for m in catalog["mechanism_classes"]
    }

    splits_applied_count = 0
    splits_dropped = []
    new_classes_created = []
    catalog_dirty = False
    for (kind, norm_name), info in split_groups.items():
        members = sorted(info["members"])
        if len(members) < MIN_GROUP_SIZE:
            splits_dropped.append(
                (info["orig_class_id"], info["name"][:60], len(members))
            )
            continue
        # Check if normalized name matches an existing class -> route members
        # to it instead of creating a duplicate (forward-protection against
        # the HC025/HC026 cross-chunk artifact).
        existing_match = (
            existing_hc_by_norm.get(norm_name)
            if kind == "HC"
            else existing_mc_by_norm.get(norm_name)
        )
        if existing_match:
            target_id = existing_match["class_id"]
            for pid in members:
                kw = {"new_hc": target_id} if kind == "HC" else {"new_mc": target_id}
                _append_path_reassignment(
                    pid,
                    source=f"final_sweep_{sweep_idx:03d}_SPLIT_OUT_existing_match",
                    rationale=f"normalized-name match to existing {target_id} "
                    f"'{existing_match['class_name'][:60]}' "
                    f"(proposed '{info['name'][:60]}')",
                    **kw,
                )
                applied += 1
            print(
                f"  SPLIT_OUT redirected: '{info['name'][:60]}' -> existing "
                f"{target_id} (normalized-name match), {len(members)} members",
                flush=True,
            )
            continue
        new_id = _next_class_id(catalog, kind)
        new_entry = {
            "class_id": new_id,
            "class_name": info["name"],
            "class_description": info["desc"] or "(no description)",
            "source": f"final_sweep_{sweep_idx:03d}_split_from_{info['orig_class_id']}",
        }
        if kind == "HC":
            new_entry["is_capability_gap"] = bool(info["is_capability_gap"])
            catalog["harm_classes"].append(new_entry)
            existing_hc_by_norm[norm_name] = new_entry  # prevent later-in-pass dup
        else:
            catalog["mechanism_classes"].append(new_entry)
            existing_mc_by_norm[norm_name] = new_entry
        catalog_dirty = True
        new_classes_created.append((new_id, info["name"], len(members)))
        for pid in members:
            kw = {"new_hc": new_id} if kind == "HC" else {"new_mc": new_id}
            _append_path_reassignment(
                pid,
                source=f"final_sweep_{sweep_idx:03d}_SPLIT_OUT_from_{info['orig_class_id']}",
                rationale=f"split-out into '{info['name'][:80]}'",
                **kw,
            )
            applied += 1
        splits_applied_count += 1
        print(
            f"  applied split: {info['orig_class_id']} -> {new_id} "
            f"('{info['name'][:60]}'), {len(members)} members",
            flush=True,
        )
    for d in splits_dropped:
        print(
            f"  DROPPED split: {d[0]} -> '{d[1]}' (only {d[2]} members <MIN)",
            flush=True,
        )

    if catalog_dirty:
        _save_active_catalog(catalog, kind_suffix=f"final_sweep_{sweep_idx:03d}")
    # Rebuild merged jsonl + refresh xlsx
    _rebuild_routing_assignments_jsonl()
    try:
        from phase2_routing_to_xlsx import build_routing_xlsx

        res = build_routing_xlsx()
        print(
            f"  xlsx refreshed: {res['n_paths']} paths, "
            f"{res['n_harm_classes']} HC + {res['n_mech_classes']} MC",
            flush=True,
        )
    except Exception as e:
        print(f"  xlsx rebuild warning: {e}", flush=True)

    summary_fp = out_dir / f"sweep_{sweep_idx:03d}_summary.json"
    M.atomic_write_json(
        summary_fp,
        {
            "sweep_idx": sweep_idx,
            "n_class_calls": n_class_calls,
            "verdict_counts": dict(verdict_counter),
            "n_applied": applied,
            "n_splits_applied": splits_applied_count,
            "n_splits_dropped": len(splits_dropped),
            "new_classes_created": [
                {"class_id": x[0], "class_name": x[1], "n_members": x[2]}
                for x in new_classes_created
            ],
            "n_total_decisions": len(all_decisions),
            "scope_class_ids": list(class_ids) if class_ids else None,
        },
    )
    state["total_final_sweeps_run"] = sweep_idx
    _save_state(state)
    # Auto-append final-sweep summary to watch_items
    extra = [
        f"**Scope:** {[c['class_id'] for _, c in classes_to_sweep]}",
        f"**Verdicts:** {dict(verdict_counter)}",
        f"**Applied:** {applied} decisions; {splits_applied_count} new splits "
        f"({len(new_classes_created)} new classes); {len(splits_dropped)} drops",
    ]
    if new_classes_created:
        extra.append("**New classes from splits:**")
        for nc in new_classes_created:
            extra.append(f"- {nc[0]}: {nc[1][:80]} (n={nc[2]})")
    _append_to_watch_items(
        "final_misfit_sweep",
        sweep_idx,
        f"Per-class sweep over {len(classes_to_sweep)} classes "
        f"({n_class_calls} chunks). {applied} reassignments + "
        f"{splits_applied_count} new-class splits APPLIED in-run.",
        extra_lines=extra,
    )
    _append_step_log(
        step_type="final_misfit_sweep",
        summary=f"Per-class sweep over {len(classes_to_sweep)} classes "
        f"({n_class_calls} chunks); {applied} reassignments + "
        f"{splits_applied_count} new-class splits.",
        extra={
            "sweep_idx": sweep_idx,
            "classes_swept": classes_to_sweep,
            "decisions": len(all_decisions),
            "verdicts": dict(verdict_counter),
            "applied": applied,
            "new_classes_created": len(new_classes_created),
            "splits_dropped": len(splits_dropped),
        },
    )
    print(f"\n=== Final misfit sweep #{sweep_idx} complete ===", flush=True)
    print(f"  class-chunk calls: {n_class_calls}", flush=True)
    print(f"  total decisions: {len(all_decisions)}", flush=True)
    print(f"  verdicts: {dict(verdict_counter)}", flush=True)
    print(f"  applied to jsonl: {applied}", flush=True)
    print(
        f"  new classes created via SPLIT_OUT: {len(new_classes_created)}", flush=True
    )
    print(f"  wrote {summary_fp.name}", flush=True)
    print("  appended summary to watch_items", flush=True)


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=[
            "opus_routing",
            "opus_routing_parallel",
            "consolidation",
            "final_audit",
            "axes_review",
            "misfit_review",
            "final_misfit_sweep",
            "backfill_oos",
        ],
        required=True,
    )
    ap.add_argument(
        "--n-batches",
        type=int,
        default=None,
        help="For opus_routing/_parallel: max batches this invocation; default = all remaining",
    )
    ap.add_argument(
        "--n-workers",
        type=int,
        default=5,
        help="For opus_routing_parallel: concurrent workers per group (default 5)",
    )
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument(
        "--sample-n",
        type=int,
        default=50,
        help="For axes_review: sample size; default 50",
    )
    ap.add_argument(
        "--classes",
        type=str,
        default=None,
        help="For final_misfit_sweep: comma-separated class IDs to limit "
        "scope, e.g. 'HC002,HC008,MC015'. Default = sweep all classes.",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="For final_misfit_sweep: members per Opus chunk. Default 100 "
        "(unchunked for classes up to n=100, better for sub-channel "
        "coherence). Decrease only if class is very large.",
    )
    ap.add_argument(
        "--scope",
        type=str,
        default="all",
        choices=["all", "self_flagged", "unassigned", "oversize"],
        help="For misfit_review: candidate scope. 'self_flagged' = R1 "
        "(low-fit + heuristic + reassign_pending only). 'unassigned' = R2 "
        "(unassigned-pool audit). 'oversize' = legacy. 'all' = legacy default.",
    )
    ap.add_argument(
        "--include-previously-held",
        action="store_true",
        help="For misfit_review: by default, paths HOLDed in any prior "
        "misfit_review (and not re-flagged since) are excluded — the "
        "no-rework filter. Pass this flag to re-include them (e.g. "
        "after a catalog overhaul that invalidates prior verdicts).",
    )
    args = ap.parse_args()

    if args.mode == "opus_routing":
        run_opus_routing(n_batches=args.n_batches, batch_size=args.batch_size)
    elif args.mode == "opus_routing_parallel":
        run_opus_routing_parallel(
            n_batches=args.n_batches,
            n_workers=args.n_workers,
            batch_size=args.batch_size,
        )
    elif args.mode == "consolidation":
        run_consolidation()
    elif args.mode == "final_audit":
        run_final_audit()
    elif args.mode == "axes_review":
        run_axes_review(sample_n=args.sample_n)
    elif args.mode == "misfit_review":
        run_misfit_review(
            scope=args.scope,
            exclude_previously_reviewed=not args.include_previously_held,
        )
    elif args.mode == "final_misfit_sweep":
        cls = [c.strip() for c in args.classes.split(",")] if args.classes else None
        run_final_misfit_sweep(chunk_size=args.chunk_size, class_ids=cls)
    elif args.mode == "backfill_oos":
        run_backfill_oos()


if __name__ == "__main__":
    main()
