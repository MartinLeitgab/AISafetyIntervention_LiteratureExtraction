"""
phase2_step4_phase2_doublet_llm_grouping.py — Phase 2 doublet LLM grouping (§19.13)

Path-level LLM grouping over all 8,954 EDGE-only hopwise VPN paths. Each path is
grouped TWICE in one call:
  - RISK GROUP    (based on the risk-node mechanism)
  - MECHANISM GROUP (based on body+intervention as one unit)

Downstream analyses (run separately):
  (a) Few-to-many MECHANISM->RISK: which mechanism groups address many risk groups
      -> transferable mechanisms (highest-value finding).
  (b) Few-to-many RISK->MECHANISM: which risk groups are addressed by many
      mechanism groups -> well-covered vs under-served risks (gap analysis).
  Plus full RG x MG matrix; 1-1 density is trivially the matrix cells.

Distinction vs subtype-level Pass B (phase2_step4_phase2_full_vpn_llm_naming.py):
  - Per-path grouping, not per-node.
  - No body-subtype-level catalogs. Body subtypes are continuum from risk-leaning
    to intervention-leaning; treat body+intervention as one mechanism unit.
  - Algorithmic baseline = frozenset+Jaccard memberships at
    step4_finalanalysis/step4_cluster_tables/frozenset_group_memberships_pareto_edge_only_all.csv
    Direct comparison axis for the paper's methodological demonstration.

Modes (all use claude-opus-4-7):
  --mode seed     200-path sample -> initial catalog of (RG..., MG...). No assignments.
                  Cost: ~200k tokens, ~5 min wall.
  --mode review   Same 200 paths -> assign to seed catalog (NO new groups in review).
                  Sanity check that the seed catalog covers its own training sample.
                  Cost: ~120k tokens, ~5 min wall.
  --mode smoke    Fresh 50 paths -> assign with NEW+coherence allowed.
                  Validates Pass B prompt before full launch.
                  Cost: ~70k tokens, ~5 min wall.
  --mode full     All 8,954 paths in batches of 50 with idempotent resume.
                  NEW groups + coherence updates allowed each batch.
                  Cost: ~8M tokens, ~12-15h wall split across sessions.

Outputs:
  phase2_doublet_seed_catalog.json                seed-gen output (initial RG+MG catalog)
  phase2_doublet_review.json                      seed-sample review assignments
  phase2_doublet_batches/batch_NNNN.json          per-batch atomic saves (Pass B)
  phase2_doublet_active_catalog.json              latest catalog (updated each batch)
  phase2_doublet_assignments.jsonl                merged per-path assignments
  phase2_doublet_coherence_log.jsonl              per-batch coherence updates
  phase2_doublet_summary.json                     final counts + group stats

Usage:
  python phase2_step4_phase2_doublet_llm_grouping.py --mode seed
  python phase2_step4_phase2_doublet_llm_grouping.py --mode review
  python phase2_step4_phase2_doublet_llm_grouping.py --mode smoke
  python phase2_step4_phase2_doublet_llm_grouping.py --mode full
"""

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("CLAUDE_CLI_TIMEOUT_SEC", "3600")

SHIM_DIR = Path("C:/Users/malei/0_project_work/0_domain_finder/knowledge_pipeline/src")
sys.path.insert(0, str(SHIM_DIR))
from claude_cli_shim import ClaudeCLI  # noqa: E402

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
PATHS_FILE = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"

# ---- Configuration ----
SEED_SAMPLE_SIZE = 200
SMOKE_SAMPLE_SIZE = 50
BATCH_SIZE = 50
DESC_TRUNCATE_CHARS = 300  # truncate each node description to ~75 tokens
SEED_TARGET_RG = 30  # seed-gen target: ~30 risk groups
SEED_TARGET_MG = 80  # seed-gen target: ~80 mechanism groups
MIN_GROUP_SIZE = (
    3  # propose new group only if >= 3 paths expected to fit (risk + mechanism)
)
SEED_RNG_SEED = 20260514
SMOKE_RNG_SEED = 20260515
BATCH_ORDER_RNG_SEED = 20260516

MAX_RETRIES_PER_BATCH = 1
SEED_MAX_TOKENS = 16384
ASSIGN_MAX_TOKENS = 16384

# Subtype role explanations -- shared with the subtype-level naming script
SUBTYPE_DEFINITIONS = """\
Each path was extracted by an LLM following a seven-step causal-interventional
reasoning chain. The CANONICAL order of roles is:

  risk                  — the failure mode / harm the path addresses
  problem_analysis      — why the risk arises (mechanistic decomposition)
  theoretical_insight   — formal / conceptual framing supporting an intervention
  design_rationale      — high-level intervention strategy
  implementation_mech   — concrete technical mechanism realising the rationale
  validation_evidence   — what evidence supports the mechanism
  intervention          — the proposed remedy

The risk node is ALWAYS first and the intervention node is ALWAYS last. The
body nodes (pa/ti/dr/im/va labels in the input) appear between them in the
ORDER the extraction LLM emitted them — this is OFTEN but NOT ALWAYS the
canonical order above. Out-of-canonical-order body sequences are still
information-bearing about the paper's actual reasoning flow; treat them as
valid input, not defects to correct or ignore. NOT every path contains every
body role; lengths vary.

For MECHANISM GROUP assignment: treat the body (all pa/ti/dr/im/va nodes,
regardless of their order in the path) PLUS the intervention as ONE mechanism
unit. Do NOT subdivide mechanism groups by body role.
"""


def atomic_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:6]}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ").replace("  ", " ")
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "..."


def load_paths_and_attrs():
    """Load 8,954 hopwise edge-only paths + node attributes.
    Returns (paths: list[dict with path_id], node_attrs: dict[int->dict]).
    """
    print(f"loading paths from {PATHS_FILE.name} ...")
    paths = []
    with open(PATHS_FILE, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["path_id"] = f"path_{lineno:05d}"
            paths.append(d)
    print(f"  loaded {len(paths)} paths")

    print("loading graph_node_attributes.pkl ...")
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  loaded {len(node_attrs)} nodes")
    return paths, node_attrs


def fmt_path(path_rec, node_attrs):
    """Render one path as compact LLM input. Node NAMES are not truncated (full names);
    only descriptions are truncated to DESC_TRUNCATE_CHARS.
    Layout:
        [path_00042]
        risk: <full name> -- <desc trunc>
        body: pa | <full name> -- <desc>
        body: ti | <full name> -- <desc>
        ...
        intervention: <full name> -- <desc>
    """
    nodes = path_rec["path"]
    cats = path_rec["categories"]
    lines = [f"[{path_rec['path_id']}]"]
    SUBTYPE_SHORT = {
        "problem_analysis": "pa",
        "theoretical_insight": "ti",
        "design_rationale": "dr",
        "implementation_mechanism": "im",
        "validation_evidence": "va",
    }
    for nid, cat in zip(nodes, cats):
        attrs = node_attrs.get(int(nid)) or node_attrs.get(nid) or {}
        name = (attrs.get("name", "?") or "").strip().replace("\n", " ")
        desc = truncate(attrs.get("description", ""), DESC_TRUNCATE_CHARS)
        if cat == "risk":
            label = "risk"
        elif cat == "intervention":
            label = "interv"
        else:
            sub = attrs.get("subtype") or attrs.get("concept_category") or "body"
            label = f"body[{SUBTYPE_SHORT.get(sub, sub[:2])}]"
        lines.append(f"  {label}: {name} -- {desc}")
    return "\n".join(lines)


def fmt_catalog_compact(rg_list, mg_list):
    """Render the active catalog compactly for inclusion in each Pass B batch.
    Format:
      RISK GROUPS (n=...):
        RG001  <name> | <1-sentence desc>
        ...
      MECHANISM GROUPS (n=...):
        MG001  <name> | <1-sentence desc>
        ...
    """
    out = [f"RISK GROUPS (n={len(rg_list)}):"]
    for g in rg_list:
        out.append(
            f"  {g['group_id']}  {g['group_name']} | {truncate(g['group_description'], 180)}"
        )
    out.append(f"\nMECHANISM GROUPS (n={len(mg_list)}):")
    for g in mg_list:
        out.append(
            f"  {g['group_id']}  {g['group_name']} | {truncate(g['group_description'], 180)}"
        )
    return "\n".join(out)


# ============================================================
# Prompts
# ============================================================


def make_seed_prompt(paths, node_attrs, sentinel):
    """Seed-gen: input N paths, produce initial RG+MG catalog AND per-path assignments
    in ONE combined output. The LLM proposes groups (with IDs it picks) and assigns
    EVERY input path to one risk_group_id + one mechanism_group_id.
    """
    body = "\n\n".join(fmt_path(p, node_attrs) for p in paths)
    n = len(paths)
    return f"""You are building an AI-safety research path catalog. Each path is a
risk -> body -> intervention chain extracted from one paper.

Your task has TWO parts in one output:
  (1) Produce the INITIAL CATALOG of two parallel group taxonomies (defined below).
  (2) Assign EVERY one of the {n} input paths to one risk_group + one mechanism_group
      from the catalog you just produced. NO path may be missing from the assignments list.

Two parallel taxonomies:
  - RISK GROUPS        — collect paths whose RISK NODE describes the same kind of failure mode.
                         Two paths share a risk group when they describe the SAME causal
                         mechanism by which the AI system could cause harm — even if surface
                         vocabulary differs.
  - MECHANISM GROUPS   — collect paths whose BODY+INTERVENTION together describe the same
                         kind of intervention strategy. Two paths share a mechanism group
                         when their intervention reduces risk via the SAME causal pathway
                         (regardless of which risk it addresses).

DOWNSTREAM USE — name and bound groups so a reader can read these two analyses
straight from the group names:
  (a) Which mechanism groups address many distinct risk groups
      -> TRANSFERABLE MECHANISMS (highest-value finding).
  (b) Which risk groups are addressed by many distinct mechanism groups
      -> WELL-COVERED RISKS vs UNDER-SERVED RISKS (gap analysis).

A group whose name does not predict its contents in those two analyses is a bad group.

============================================================
PATH ROLES CONTEXT
============================================================

{SUBTYPE_DEFINITIONS}

============================================================
GRANULARITY GUIDANCE
============================================================

Target ~{SEED_TARGET_RG} risk groups, ~{SEED_TARGET_MG} mechanism groups across the full corpus.
This is the seed catalog from {n} sample paths; later batches will grow + refine it.
Propose a group only if you expect >= {MIN_GROUP_SIZE} paths total (across full corpus) will fit it.
For risks specifically: do NOT collapse capability-elicitation risks (e.g. underelicitation,
sandbagging detection) with alignment/misuse risks — they belong in distinct risk groups so
the paper can separate them.

ID conventions: risk_group ids are "RG001", "RG002", ... (zero-padded to 3 digits, sequential
from RG001). mechanism_group ids are "MG001", "MG002", ... Use ONLY ids you define in this
output's risk_groups / mechanism_groups arrays.

============================================================
INPUT PATHS ({n} paths)
============================================================

{body}

============================================================
OUTPUT FORMAT (STRICT — validation will reject malformed responses)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences. No commentary.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.
- The `assignments` array MUST contain exactly {n} entries — one per input path. Validation
  will reject any output where len(assignments) != {n} or any path_id is missing.

Schema:

{{
  "risk_groups": [
    {{"group_id": "RG001", "group_name": "<short distinctive name>",
      "group_description": "<1-2 sentences identifying the causal failure mode>"}},
    ...
  ],
  "mechanism_groups": [
    {{"group_id": "MG001", "group_name": "<short distinctive name>",
      "group_description": "<1-2 sentences identifying the intervention strategy / causal pathway>"}},
    ...
  ],
  "assignments": [
    {{"path_id": "path_00042",
      "risk_group_id": "RG017",
      "mechanism_group_id": "MG023"}},
    ... (one entry for EVERY input path; no path may be missing)
  ]
}}END_SENTINEL_{sentinel}

Produce the catalog AND assignments now."""


def make_assign_prompt(
    paths,
    node_attrs,
    rg_list,
    mg_list,
    sentinel,
    allow_new=True,
    allow_coherence=True,
    allow_unassigned=False,
):
    """Assign each path to one RG + one MG from the active catalog.
    If allow_new=True, LLM may propose new groups (use 'new' decision).
    If allow_coherence=True, LLM emits per-group coherence updates at end.
    If allow_unassigned=True, LLM may emit {"unassigned": "<reason>"} for paths
    whose risk OR mechanism does NOT fit any existing group. Used by Sonnet
    Pass B (read-only catalog) so unfittable paths are explicitly surfaced for
    Opus REVIEW_C, rather than force-fit to a wrong group.
    Note: allow_new and allow_unassigned are typically mutually exclusive — Sonnet
    Pass B uses (allow_new=False, allow_unassigned=True); seed-gen uses
    (allow_new=True, allow_unassigned=False).
    """
    body = "\n\n".join(fmt_path(p, node_attrs) for p in paths)
    n = len(paths)
    catalog = fmt_catalog_compact(rg_list, mg_list)
    options = []
    if allow_new:
        options.append('{"new": {"group_name": "...", "group_description": "..."}}')
    if allow_unassigned:
        options.append('{"unassigned": "<short reason - 1 sentence>"}')
    extra_options = (" OR " + " OR ".join(options)) if options else ""
    new_clause = (
        ""
        if not allow_new
        else f"""
  - For risk_group OR mechanism_group, you may instead emit:
      "new": {{"group_name": "<short name>", "group_description": "<1-2 sentences>"}}
    Use ONLY when the path's risk/mechanism is clearly absent from the catalog AND
    you expect >= {MIN_GROUP_SIZE} paths will fit the new group. Re-use the same new
    group_name verbatim across multiple paths in this batch if they belong together.
"""
    )
    unassigned_clause = (
        ""
        if not allow_unassigned
        else """
  - For risk_group OR mechanism_group, you may instead emit:
      "unassigned": "<short reason - 1 sentence on why no existing group fits>"
    Use ONLY when the path's risk/mechanism is clearly absent from the catalog AND
    you would otherwise have to force-fit it to a wrong group. The unassigned rows
    will be reviewed by a higher-capability model in a separate pass that CAN
    propose new groups; do NOT propose new groups here.
"""
    )
    coherence_trailing_comma = "," if allow_coherence else ""
    coherence_field = '  "coherence_updates": [ ... ]' if allow_coherence else ""
    coherence_clause = (
        ""
        if not allow_coherence
        else """
After the path-assignment list, also emit:
  "coherence_updates": [
    {"group_id": "RG017", "coherence": "high"|"medium"|"low"|"fragmenting",
     "split_hint": "<if fragmenting, 1 sentence on how it would split; else null>"},
    ...
  ]
ONLY include coherence updates for groups you actually assigned paths to in this batch.
Mark `coherence` as "fragmenting" when adding this batch's paths reveals the group is
subsuming distinct mechanisms that should be separated.
"""
    )

    return f"""You are assigning AI-safety research paths to an evolving doublet catalog.

For EACH input path (one entry per path; no path may be missing; no duplicates), output:
  - "path_id": "path_XXXXX"  (verbatim from the path header)
  - "risk_group":      either {{"existing": "RG###"}}{extra_options}
  - "mechanism_group": either {{"existing": "MG###"}}{extra_options}
{new_clause}{unassigned_clause}
DOWNSTREAM USE — your assignments feed two analyses:
  (a) Which mechanism groups address many distinct risk groups (transferable mechanisms).
  (b) Which risk groups are addressed by many distinct mechanism groups (gap analysis).
Assign so these two analyses produce interpretable results.
The `assignments` array MUST contain exactly {n} entries — one per input path.
{coherence_clause}
============================================================
PATH ROLES CONTEXT
============================================================

{SUBTYPE_DEFINITIONS}

============================================================
ACTIVE CATALOG
============================================================

{catalog}

============================================================
INPUT PATHS ({n} paths)
============================================================

{body}

============================================================
OUTPUT FORMAT (STRICT — validation will reject malformed responses)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences. No commentary.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.

Schema:

{{
  "assignments": [
    {{"path_id": "path_00042",
      "risk_group": {{"existing": "RG017"}},
      "mechanism_group": {{"existing": "MG023"}}}},
    {{"path_id": "path_00043",
      "risk_group": {{"new": {{"group_name": "...", "group_description": "..."}}}},
      "mechanism_group": {{"existing": "MG023"}}}}
  ]{coherence_trailing_comma}
{coherence_field}
}}END_SENTINEL_{sentinel}

Produce the assignments now."""


# ============================================================
# LLM call wrappers
# ============================================================
# Two paths:
#   call_with_validation(...)            — shim-based; subprocess.run blocking.
#                                          Use for SHORT outputs (<10k tokens).
#                                          Smoke / Pass B per-batch / review.
#   streaming_call_with_validation(...)  — direct `claude -p --output-format
#                                          stream-json`; streams output to a
#                                          partial file as tokens arrive. Use
#                                          for LARGE outputs (>10k tokens), e.g.
#                                          the combined seed-gen call.
#
# Why two paths exist:
#   The shim-based path is simpler and fine for small outputs, but it
#   subprocess.run(capture_output=True, timeout=900) which (a) loses ALL output
#   if the subprocess hangs and (b) on Windows can orphan claude.exe via the
#   CMD-wrapper timeout-propagation bug, hanging the parent indefinitely.
#   The streaming path bypasses subprocess.run entirely (uses Popen +
#   line-iteration over stdout), persists every token to disc as it arrives,
#   and has no hard timeout cliff.
# ============================================================

import subprocess  # noqa: E402

_SHIM_DIR_FOR_AUTH = SHIM_DIR  # reuse the same env-strip pattern as the shim
_AUTH_VARS_THAT_LEAK = (
    "ANTHROPIC_API_KEY",
    "anthropic_api_key",
    "ANTHROPIC_AUTH_TOKEN",
)


def _streaming_child_env():
    """Strip auth env vars so Max-plan subscription billing is used (not metered API).
    Mirrors claude_cli_shim:374-381.
    """
    leaking = [k for k in _AUTH_VARS_THAT_LEAK if os.environ.get(k)]
    if leaking:
        return {
            k: v
            for k, v in os.environ.items()
            if k not in _AUTH_VARS_THAT_LEAK
            and k not in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")
        }
    return dict(os.environ)


def _resolve_claude_bin():
    """Resolve the claude CLI invocation.

    On Windows: `npm` installs claude as `claude.cmd`. `where claude` returns
    BOTH `claude` (no extension, not directly executable from subprocess.Popen)
    and `claude.cmd`. Use cmd.exe /c with bare `claude` so Windows PATHEXT
    resolution picks the .cmd shim automatically. Avoids WinError 193
    ("not a valid Win32 application") that hits if we try to Popen the
    no-extension entry directly.

    On POSIX: invoke `claude` directly.
    """
    if os.name == "nt":
        return ["cmd.exe", "/c", "claude"]
    return ["claude"]


def streaming_claude_call(prompt, system, partial_path, model="claude-opus-4-7"):
    """Stream a single LLM call via `claude -p --output-format stream-json`.

    Writes each text delta to `partial_path` IMMEDIATELY as it arrives
    (line-buffered, flushed after each event). Even if the process is killed
    mid-stream, partial_path contains all text emitted up to that point.

    Returns (text, duration_sec, error_or_None).
      - On success: text is the full assembled assistant response, error=None.
      - On failure: text is whatever was streamed before failure, error
        describes what went wrong. partial_path on disc matches `text`.

    Live progress: `tail -f <partial_path>` shows the model output as it streams.
    """
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text("", encoding="utf-8")  # reset

    base_cmd = _resolve_claude_bin()
    cmd = base_cmd + [
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",  # required by Claude CLI for stream-json
        "--model",
        model,
        "--system-prompt",
        system,
    ]

    env = _streaming_child_env()
    t0 = time.time()
    text_parts = []
    error = None
    last_print = t0
    n_events = 0

    print(
        f"  [stream] launch: {' '.join(base_cmd)} -p stream-json model={model}",
        flush=True,
    )
    print(f"  [stream] partial file: {partial_path}", flush=True)
    print(
        f"  [stream] prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)",
        flush=True,
    )

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # line-buffered
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except Exception as e:
        error = f"Popen failed: {type(e).__name__}: {str(e)[:200]}"
        return "", time.time() - t0, error

    # Write prompt to stdin then close (signals EOF to CLI)
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except Exception as e:
        error = f"stdin write failed: {type(e).__name__}: {str(e)[:200]}"
        try:
            proc.kill()
        except Exception:
            pass
        return "", time.time() - t0, error

    # Stream events from stdout, persist text deltas to disc immediately.
    # Also write a debug log of every event type seen, to diagnose missing-text
    # situations (e.g., Sonnet emitting via a different event shape than Opus).
    debug_log_path = partial_path.with_suffix(".debug_events.jsonl")
    event_type_counts = {}
    fallback_assistant_text = None  # populated from final assistant event if seen
    try:
        debug_log = open(debug_log_path, "w", buffering=1, encoding="utf-8")
        with open(
            partial_path, "w", buffering=1, encoding="utf-8", errors="replace"
        ) as fout:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                n_events += 1
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Track event-type frequency for debug
                etype = evt.get("type", "<no_type>")
                if etype == "stream_event":
                    inner = evt.get("event", {})
                    inner_type = inner.get("type", "<no_type>")
                    if inner_type == "content_block_delta":
                        delta_type = inner.get("delta", {}).get("type", "<no_type>")
                        composite_key = f"stream_event.content_block_delta.{delta_type}"
                    else:
                        composite_key = f"stream_event.{inner_type}"
                else:
                    composite_key = etype
                event_type_counts[composite_key] = (
                    event_type_counts.get(composite_key, 0) + 1
                )
                # Write event header (NOT full body) to debug log for diagnosis
                debug_log.write(
                    json.dumps({"n": n_events, "key": composite_key, "size": len(line)})
                    + "\n"
                )
                debug_log.flush()

                # Extract text deltas from possible event shapes:
                # (a) stream_event > content_block_delta > text_delta (preferred,
                #     enabled by --include-partial-messages)
                # (b) assistant > message > content[].text (final assembled — used
                #     as fallback if (a) never emitted any text)
                delta_text = None
                if etype == "stream_event":
                    inner = evt.get("event", {})
                    if inner.get("type") == "content_block_delta":
                        d = inner.get("delta", {})
                        if d.get("type") == "text_delta":
                            delta_text = d.get("text", "")
                elif etype == "assistant":
                    # Final assembled assistant message; capture as fallback
                    msg = evt.get("message", {})
                    blocks = msg.get("content", []) if isinstance(msg, dict) else []
                    parts = []
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "text":
                            parts.append(b.get("text", ""))
                    if parts:
                        fallback_assistant_text = "".join(parts)

                if delta_text:
                    fout.write(delta_text)
                    fout.flush()
                    text_parts.append(delta_text)

                # Heartbeat every 10s so live monitoring sees activity
                now = time.time()
                if now - last_print > 10:
                    cum_chars = sum(len(p) for p in text_parts)
                    n_think = event_type_counts.get(
                        "stream_event.content_block_delta.thinking_delta", 0
                    )
                    n_text = event_type_counts.get(
                        "stream_event.content_block_delta.text_delta", 0
                    )
                    print(
                        f"  [stream] +{now - t0:.0f}s: {n_events} events "
                        f"({n_think} think, {n_text} text); "
                        f"{cum_chars} chars streamed",
                        flush=True,
                    )
                    last_print = now
        debug_log.close()
    except Exception as e:
        error = f"stream read failed: {type(e).__name__}: {str(e)[:200]}"
        try:
            proc.kill()
        except Exception:
            pass

    rc = proc.wait()
    duration = time.time() - t0
    text = "".join(text_parts)

    # Fallback: if no text deltas emitted (e.g., Sonnet via claude-code CLI not
    # using `content_block_delta` even with --include-partial-messages) but the
    # final `assistant` event was captured, use that. Write to partial file too.
    if not text and fallback_assistant_text:
        text = fallback_assistant_text
        try:
            with open(partial_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
        print(
            f"  [stream] FALLBACK: no text_delta events; used final assistant "
            f"message ({len(text)} chars)",
            flush=True,
        )

    if rc != 0 and error is None:
        try:
            stderr_text = proc.stderr.read() if proc.stderr else ""
        except Exception:
            stderr_text = ""
        error = f"claude -p exit {rc}; stderr[:500]={stderr_text[:500]}"

    # Print event-type summary so future diagnostics is easy
    if event_type_counts:
        sorted_types = sorted(event_type_counts.items(), key=lambda x: -x[1])
        summary = ", ".join(f"{k}={v}" for k, v in sorted_types[:8])
        print(f"  [stream] event types seen: {summary}", flush=True)

    print(
        f"  [stream] DONE: {len(text)} chars in {duration:.0f}s, "
        f"{n_events} events, error={error}",
        flush=True,
    )
    return text, duration, error


def call_with_validation(
    prompt, sentinel, label, max_tokens, max_retries=MAX_RETRIES_PER_BATCH
):
    client = ClaudeCLI()
    end_marker = f"END_SENTINEL_{sentinel}"
    for attempt in range(max_retries + 1):
        print(f"  [{label}] attempt {attempt + 1}/{max_retries + 1} ...", flush=True)
        t0 = time.time()
        try:
            resp = client.messages.create(
                model="claude-opus-4-7",
                system=(
                    "You produce STRICT JSON output for an AI-safety doublet "
                    "grouping pipeline. Never preamble, never use markdown fences, "
                    "always emit valid JSON, always end your output with the "
                    "requested sentinel."
                ),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            text = resp.content[0].text
            duration = time.time() - t0
            print(f"    returned {len(text)} chars in {duration:.0f}s", flush=True)
            trimmed = text.strip()
            if trimmed.startswith("{") and trimmed.endswith(end_marker):
                json_part = trimmed[: -len(end_marker)].rstrip()
                return json_part, duration, attempt + 1, None
            else:
                print(
                    f"    FAIL validation; first/last 100: {repr(trimmed[:100])} ... {repr(trimmed[-100:])}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"    FAIL shim error: {type(e).__name__}: {str(e)[:200]}", flush=True
            )
    return None, 0.0, max_retries + 1, "validation"


def streaming_call_with_validation(
    prompt,
    sentinel,
    label,
    partial_path,
    max_retries=MAX_RETRIES_PER_BATCH,
    model="claude-opus-4-7",
):
    """Streaming counterpart to call_with_validation, for large-output calls.
    Use for the combined seed-gen (~60k output tokens) where the blocking shim
    pattern is unsafe. See module-level comment for details.
    model: defaults to opus; pass "claude-sonnet-4-6" for Sonnet Pass B routing
    calls where extended-thinking is unnecessary and Sonnet's cheaper per-token
    Max-plan weight (~0.41x Opus, measured 2026-05-15) is preferred.
    """
    end_marker = f"END_SENTINEL_{sentinel}"
    system = (
        "You produce STRICT JSON output for an AI-safety doublet "
        "grouping pipeline. Never preamble, never use markdown fences, "
        "always emit valid JSON, always end your output with the "
        "requested sentinel."
    )
    for attempt in range(max_retries + 1):
        print(
            f"  [{label}] attempt {attempt + 1}/{max_retries + 1} (streaming, model={model}) ...",
            flush=True,
        )
        text, duration, error = streaming_claude_call(
            prompt, system, partial_path, model=model
        )
        if error:
            print(f"    STREAM ERROR: {error}", flush=True)
            print(
                f"    {len(text)} chars on disc at {partial_path.name} for recovery",
                flush=True,
            )
            continue
        trimmed = text.strip()
        if trimmed.startswith("{") and trimmed.endswith(end_marker):
            json_part = trimmed[: -len(end_marker)].rstrip()
            print(f"    SENTINEL OK; {len(json_part)} chars JSON payload", flush=True)
            return json_part, duration, attempt + 1, None
        else:
            print(
                f"    FAIL sentinel validation; first/last 100: "
                f"{repr(trimmed[:100])} ... {repr(trimmed[-100:])}",
                flush=True,
            )
            print(
                f"    raw stream on disc at {partial_path.name} for inspection",
                flush=True,
            )
    return None, 0.0, max_retries + 1, "validation"


# ============================================================
# Main mode handlers
# ============================================================


def run_seed(paths, node_attrs, n_paths=None):
    """Run combined seed-gen (catalog + per-path assignments).
    n_paths: optional override of SEED_SAMPLE_SIZE for smoke testing.
    """
    out = STEP1 / "phase2_doublet_seed_catalog.json"
    if out.exists():
        print(f"[idempotent skip] {out.name} already exists")
        return json.loads(out.read_text(encoding="utf-8"))

    rng = random.Random(SEED_RNG_SEED)
    target_n = n_paths if n_paths is not None else SEED_SAMPLE_SIZE
    sample = rng.sample(paths, min(target_n, len(paths)))
    smoke_tag = " (SMOKE)" if n_paths is not None and n_paths < SEED_SAMPLE_SIZE else ""
    print(
        f"\n=== SEED-GEN{smoke_tag}: {len(sample)} paths (combined catalog + assignments) ===",
        flush=True,
    )
    sentinel = uuid.uuid4().hex[:12]
    prompt = make_seed_prompt(sample, node_attrs, sentinel)
    print(f"prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)

    # Stream output to disc as it arrives. If the call dies mid-stream,
    # the partial file is on disc for recovery.
    partial_path = STEP1 / "phase2_doublet_seed_partial.txt"
    json_part, dur, _, err = streaming_call_with_validation(
        prompt, sentinel, "seed-gen", partial_path
    )
    if err or not json_part:
        print(f"SEED-GEN FAILED ({err})", flush=True)
        print(f"  partial output preserved at: {partial_path}", flush=True)
        print(
            "  inspect manually; partial JSON may be recoverable by hand-truncation",
            flush=True,
        )
        return None
    try:
        parsed = json.loads(json_part)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return None

    rg_list = parsed.get("risk_groups", [])
    mg_list = parsed.get("mechanism_groups", [])
    assignments = parsed.get(
        "assignments", []
    )  # [{path_id, risk_group_id, mechanism_group_id}, ...]

    # Validation: every input path should appear exactly once in assignments
    expected_pids = {p["path_id"] for p in sample}
    assigned_pids = {a.get("path_id") for a in assignments}
    missing = expected_pids - assigned_pids
    extra = assigned_pids - expected_pids
    if missing:
        print(
            f"  WARNING: {len(missing)} input paths missing from assignments (e.g. {list(missing)[:3]})"
        )
    if extra:
        print(
            f"  WARNING: {len(extra)} extra/unknown path_ids in assignments (e.g. {list(extra)[:3]})"
        )

    obj = {
        "n_input_paths": len(sample),
        "input_path_ids": [p["path_id"] for p in sample],
        "n_risk_groups": len(rg_list),
        "n_mechanism_groups": len(mg_list),
        "n_assignments": len(assignments),
        "risk_groups": rg_list,
        "mechanism_groups": mg_list,
        "assignments": assignments,
    }
    atomic_write_json(out, obj)
    print(
        f"\nSEED CATALOG: {len(rg_list)} risk groups, {len(mg_list)} mechanism groups, "
        f"{len(assignments)} per-path assignments"
    )
    print(f"wrote {out.name}")

    rebuild_merged_assignments_jsonl()
    # Auto-build xlsx so user can review in spreadsheet (jsonl format is useless for review).
    try:
        from phase2_step4_phase2_doublet_to_xlsx import build_xlsx_from_disc

        build_xlsx_from_disc()
    except PermissionError as e:
        print(f"  xlsx WRITE BLOCKED (file open in Excel?): {e}")
    except Exception as e:
        print(f"  xlsx build FAILED ({type(e).__name__}): {e}")
    return obj


def run_review(paths, node_attrs):
    seed = json.loads(
        (STEP1 / "phase2_doublet_seed_catalog.json").read_text(encoding="utf-8")
    )
    out = STEP1 / "phase2_doublet_review.json"
    if out.exists():
        print(f"[idempotent skip] {out.name} already exists")
        return

    sample_ids = set(seed["input_path_ids"])
    sample = [p for p in paths if p["path_id"] in sample_ids]
    print(
        f"\n=== SEED-SAMPLE REVIEW: {len(sample)} paths, "
        f"catalog={seed['n_risk_groups']} RG + {seed['n_mechanism_groups']} MG ==="
    )

    sentinel = uuid.uuid4().hex[:12]
    prompt = make_assign_prompt(
        sample,
        node_attrs,
        seed["risk_groups"],
        seed["mechanism_groups"],
        sentinel,
        allow_new=False,
        allow_coherence=False,
    )
    print(f"prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

    json_part, dur, _, err = call_with_validation(
        prompt, sentinel, "review", ASSIGN_MAX_TOKENS
    )
    if err or not json_part:
        print(f"REVIEW FAILED ({err})")
        return
    parsed = json.loads(json_part)
    n_assigned = len(parsed.get("assignments", []))
    atomic_write_json(
        out,
        {
            "n_input_paths": len(sample),
            "n_assigned": n_assigned,
            "assignments": parsed.get("assignments", []),
        },
    )
    print(f"REVIEW DONE: {n_assigned}/{len(sample)} paths assigned to seed catalog")


def run_smoke(paths, node_attrs):
    seed = json.loads(
        (STEP1 / "phase2_doublet_seed_catalog.json").read_text(encoding="utf-8")
    )
    out = STEP1 / "phase2_doublet_smoke.json"
    if out.exists():
        print(f"[idempotent skip] {out.name} already exists")
        return

    rng = random.Random(SMOKE_RNG_SEED)
    seed_ids = set(seed["input_path_ids"])
    pool = [p for p in paths if p["path_id"] not in seed_ids]
    sample = rng.sample(pool, min(SMOKE_SAMPLE_SIZE, len(pool)))
    print(
        f"\n=== SMOKE: {len(sample)} fresh paths, catalog="
        f"{seed['n_risk_groups']} RG + {seed['n_mechanism_groups']} MG ==="
    )

    sentinel = uuid.uuid4().hex[:12]
    prompt = make_assign_prompt(
        sample,
        node_attrs,
        seed["risk_groups"],
        seed["mechanism_groups"],
        sentinel,
        allow_new=True,
        allow_coherence=True,
    )
    print(f"prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

    json_part, dur, _, err = call_with_validation(
        prompt, sentinel, "smoke", ASSIGN_MAX_TOKENS
    )
    if err or not json_part:
        print(f"SMOKE FAILED ({err})")
        return
    parsed = json.loads(json_part)
    n_assigned = len(parsed.get("assignments", []))
    n_coherence = len(parsed.get("coherence_updates", []))
    # Resolve assignments to canonical IDs (mint new group IDs for any "new" proposals);
    # smoke does NOT update the seed catalog on disc — it's a one-off test. The mutated
    # rg_list/mg_list are scratch copies for resolving the smoke output only.
    rg_scratch = list(seed["risk_groups"])
    mg_scratch = list(seed["mechanism_groups"])
    rg_scratch, mg_scratch, resolved = resolve_assignments_and_update_catalog(
        rg_scratch, mg_scratch, parsed
    )
    atomic_write_json(
        out,
        {
            "n_input_paths": len(sample),
            "n_assigned": n_assigned,
            "n_coherence_updates": n_coherence,
            "assignments": parsed.get("assignments", []),  # raw LLM output
            "resolved_assignments": resolved,  # canonical IDs (scratch catalog)
            "scratch_new_rg_count": len(rg_scratch) - len(seed["risk_groups"]),
            "scratch_new_mg_count": len(mg_scratch) - len(seed["mechanism_groups"]),
            "coherence_updates": parsed.get("coherence_updates", []),
        },
    )
    rebuild_merged_assignments_jsonl()
    try:
        from phase2_step4_phase2_doublet_to_xlsx import build_xlsx_from_disc

        build_xlsx_from_disc()
    except PermissionError as e:
        print(f"  xlsx WRITE BLOCKED (file open in Excel?): {e}")
    except Exception as e:
        print(f"  xlsx build FAILED ({type(e).__name__}): {e}")
    print(f"SMOKE DONE: {n_assigned} assignments, {n_coherence} coherence updates")


def resolve_assignments_and_update_catalog(rg_list, mg_list, parsed_batch):
    """Apply 'new' group proposals (idempotent on group_name) AND return per-path
    assignments resolved to canonical IDs.

    Input per-path assignment shape (Pass B / smoke):
      {"path_id": "...",
       "risk_group":      {"existing": "RG017"} OR {"new": {"group_name": "...", "group_description": "..."}},
       "mechanism_group": {"existing": "MG023"} OR {"new": {"group_name": "...", "group_description": "..."}}}

    Output resolved-assignment shape (canonical):
      {"path_id": "...", "risk_group_id": "RG017", "mechanism_group_id": "MG023"}
    """
    rg_by_name = {g["group_name"]: g for g in rg_list}
    mg_by_name = {g["group_name"]: g for g in mg_list}
    rg_ids = {g["group_id"] for g in rg_list}
    mg_ids = {g["group_id"] for g in mg_list}
    next_rg_idx = max([int(g["group_id"][2:]) for g in rg_list], default=0)
    next_mg_idx = max([int(g["group_id"][2:]) for g in mg_list], default=0)

    def resolve_side(decision, side_kind):
        """side_kind is 'risk' or 'mechanism'. Return canonical group_id string or None."""
        nonlocal next_rg_idx, next_mg_idx
        if not isinstance(decision, dict):
            return None
        if "existing" in decision:
            gid = decision["existing"]
            valid_ids = rg_ids if side_kind == "risk" else mg_ids
            return gid if gid in valid_ids else None
        if "new" in decision:
            new_g = decision["new"]
            gname = (new_g.get("group_name") or "").strip()
            if not gname:
                return None
            by_name = rg_by_name if side_kind == "risk" else mg_by_name
            if gname in by_name:
                return by_name[gname]["group_id"]
            if side_kind == "risk":
                next_rg_idx += 1
                new_id = f"RG{next_rg_idx:03d}"
                entry = {
                    "group_id": new_id,
                    "group_name": gname,
                    "group_description": new_g.get("group_description", ""),
                }
                rg_list.append(entry)
                rg_by_name[gname] = entry
                rg_ids.add(new_id)
            else:
                next_mg_idx += 1
                new_id = f"MG{next_mg_idx:03d}"
                entry = {
                    "group_id": new_id,
                    "group_name": gname,
                    "group_description": new_g.get("group_description", ""),
                }
                mg_list.append(entry)
                mg_by_name[gname] = entry
                mg_ids.add(new_id)
            return new_id
        return None

    resolved = []
    for a in parsed_batch.get("assignments", []):
        pid = a.get("path_id")
        if not pid:
            continue
        rg_id = resolve_side(a.get("risk_group", {}), "risk")
        mg_id = resolve_side(a.get("mechanism_group", {}), "mechanism")
        resolved.append(
            {"path_id": pid, "risk_group_id": rg_id, "mechanism_group_id": mg_id}
        )
    return rg_list, mg_list, resolved


def _resolve_remap_transitive(remap_dict):
    """A group_remap entry RG009 -> RG017 may chain (RG017 -> RG024). Resolve the final
    target for every key; returns a new dict with no chains."""
    final = {}
    for k in remap_dict:
        seen = set()
        cur = k
        while cur in remap_dict and cur not in seen:
            seen.add(cur)
            cur = remap_dict[cur]
        final[k] = cur
    return final


def _load_group_remap():
    """Read phase2_doublet_group_remap.json if it exists. Returns (rg_remap, mg_remap)
    after transitive resolution. Empty dicts when file missing."""
    fp = STEP1 / "phase2_doublet_group_remap.json"
    if not fp.exists():
        return {}, {}
    d = json.loads(fp.read_text(encoding="utf-8"))
    return (
        _resolve_remap_transitive(d.get("rg", {}) or {}),
        _resolve_remap_transitive(d.get("mg", {}) or {}),
    )


def _load_path_remap():
    """Read phase2_doublet_path_remap.json if it exists. Returns dict path_id ->
    {risk_group_id?, mechanism_group_id?}. Used after REVIEW_B splits to override
    specific paths' assignments without rewriting batch files."""
    fp = STEP1 / "phase2_doublet_path_remap.json"
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def rebuild_merged_assignments_jsonl():
    """Rebuild phase2_doublet_assignments.jsonl from all sources on disc:
      - seed catalog assignments (source=seed)
      - smoke assignments (source=smoke, uses resolved IDs)
      - Pass B per-batch resolved_assignments (source=batch_NNNN / passb_batch_NNNN)
      - Opus REVIEW_C new-assignment batches (source=review_c_NNN)
    Applies on read:
      - group_remap (RG/MG merges from REVIEW_A and REVIEW_B)
      - path_remap (per-path overrides from REVIEW_B splits)
    Idempotent: always overwrites the merged file. Safe to call any time.
    """
    out = STEP1 / "phase2_doublet_assignments.jsonl"
    rg_remap, mg_remap = _load_group_remap()
    path_remap = _load_path_remap()

    def remap_row(r):
        """Apply group_remap and path_remap to a row in place; return the row."""
        rg = r.get("risk_group_id")
        mg = r.get("mechanism_group_id")
        if rg in rg_remap:
            r["risk_group_id"] = rg_remap[rg]
        if mg in mg_remap:
            r["mechanism_group_id"] = mg_remap[mg]
        pid = r.get("path_id")
        if pid and pid in path_remap:
            override = path_remap[pid]
            if "risk_group_id" in override:
                r["risk_group_id"] = override["risk_group_id"]
            if "mechanism_group_id" in override:
                r["mechanism_group_id"] = override["mechanism_group_id"]
        return r

    rows = []
    # Seed
    seed_fp = STEP1 / "phase2_doublet_seed_catalog.json"
    if seed_fp.exists():
        seed = json.loads(seed_fp.read_text(encoding="utf-8"))
        for a in seed.get("assignments", []):
            rows.append(
                remap_row(
                    {
                        "path_id": a.get("path_id"),
                        "risk_group_id": a.get("risk_group_id"),
                        "mechanism_group_id": a.get("mechanism_group_id"),
                        "source": "seed",
                    }
                )
            )
    # Smoke (uses resolved IDs persisted in smoke output if available)
    smoke_fp = STEP1 / "phase2_doublet_smoke.json"
    if smoke_fp.exists():
        smoke = json.loads(smoke_fp.read_text(encoding="utf-8"))
        for a in smoke.get("resolved_assignments", []):
            rows.append(remap_row({**a, "source": "smoke"}))
    # Pass B per-batch resolved assignments (legacy Opus pass_b — kept for resume)
    batch_dir = STEP1 / "phase2_doublet_batches"
    if batch_dir.exists():
        for bf in sorted(batch_dir.glob("batch_*.json")):
            d = json.loads(bf.read_text(encoding="utf-8"))
            src = (
                f"batch_{d.get('batch_idx', '?'):04d}"
                if isinstance(d.get("batch_idx"), int)
                else bf.stem
            )
            for a in d.get("resolved_assignments", []):
                rows.append(remap_row({**a, "source": src}))
    # Pass B per-batch resolved assignments (Sonnet pass_b_sonnet — current canonical)
    passb_dir = STEP1 / "phase2_doublet_passb_batches"
    if passb_dir.exists():
        for bf in sorted(passb_dir.glob("batch_*.json")):
            d = json.loads(bf.read_text(encoding="utf-8"))
            src = (
                f"passb_batch_{d.get('batch_idx', '?'):04d}"
                if isinstance(d.get("batch_idx"), int)
                else bf.stem
            )
            for a in d.get("resolved_assignments", []):
                rows.append(remap_row({**a, "source": src}))
    # Opus REVIEW_C new-assignment batches (path_id -> RG/MG, possibly new groups)
    review_dir = STEP1 / "phase2_doublet_opus_reviews"
    if review_dir.exists():
        for rcf in sorted(review_dir.glob("review_c_*.json")):
            d = json.loads(rcf.read_text(encoding="utf-8"))
            src = rcf.stem
            for a in d.get("resolved_assignments", []):
                rows.append(remap_row({**a, "source": src}))
    # Atomic write
    tmp = out.with_suffix(out.suffix + f".tmp.{uuid.uuid4().hex[:6]}")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    return out, len(rows)


def run_full(paths, node_attrs):
    # Lazy import: xlsx builder is invoked after each batch so the user sees the
    # spreadsheet refresh continuously. Import here (not at module level) to keep
    # cold-start fast for seed/review/smoke modes.
    from phase2_step4_phase2_doublet_to_xlsx import (
        build_xlsx_from_disc,
        build_edge_lookup,
        load_paths_indexed,
    )

    seed_path = STEP1 / "phase2_doublet_seed_catalog.json"
    if not seed_path.exists():
        raise FileNotFoundError(
            f"\nERROR: required artifact missing.\n"
            f"  Expected: {seed_path}\n"
            f"  Produced by: `python phase2_step4_phase2_doublet_llm_grouping.py --mode seed`\n"
            f"  Pass B (full) requires the seed catalog to exist; this script does NOT bootstrap\n"
            f"  the catalog from scratch.\n"
        )
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    rg_list = list(seed["risk_groups"])
    mg_list = list(seed["mechanism_groups"])

    # Pre-build edge_lookup + paths_idx ONCE for the xlsx builder; reused every batch.
    print("loading edge data for xlsx builder ...")
    with open(STEP1 / "graph_edge_data.pkl", "rb") as f:
        edge_data = pickle.load(f)
    edge_lookup = build_edge_lookup(edge_data)
    print(f"  {len(edge_lookup)} EDGE-type edges indexed")
    paths_idx = load_paths_indexed()
    print(f"  {len(paths_idx)} paths indexed for xlsx builder")

    # Load active catalog if it exists (resume case)
    active_path = STEP1 / "phase2_doublet_active_catalog.json"
    if active_path.exists():
        active = json.loads(active_path.read_text(encoding="utf-8"))
        rg_list = active["risk_groups"]
        mg_list = active["mechanism_groups"]
        print(f"resumed active catalog: {len(rg_list)} RG + {len(mg_list)} MG")

    # Identify already-assigned paths from prior batches
    batch_dir = STEP1 / "phase2_doublet_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    done_path_ids = set()
    for bf in sorted(batch_dir.glob("batch_*.json")):
        d = json.loads(bf.read_text(encoding="utf-8"))
        for a in d.get("assignments", []):
            done_path_ids.add(a.get("path_id"))
    print(f"already-decided paths: {len(done_path_ids)}")

    # Remaining
    remaining = [p for p in paths if p["path_id"] not in done_path_ids]
    print(f"remaining: {len(remaining)} of {len(paths)}")

    if not remaining:
        print("All paths done. Run consolidate step separately.")
        return

    rng = random.Random(BATCH_ORDER_RNG_SEED)
    rng.shuffle(remaining)

    n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    existing_batches = sorted(batch_dir.glob("batch_*.json"))
    next_batch_idx = (
        max(
            [int(re.search(r"batch_(\d+)", f.name).group(1)) for f in existing_batches],
            default=-1,
        )
    ) + 1
    print(f"will run {n_batches} new batches starting at batch_{next_batch_idx:04d}")

    coherence_log = []
    coherence_log_path = STEP1 / "phase2_doublet_coherence_log.jsonl"
    if coherence_log_path.exists():
        coherence_log = [
            json.loads(line)
            for line in coherence_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    for bi in range(n_batches):
        batch_idx = next_batch_idx + bi
        batch = remaining[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        print(
            f"\n--- batch_{batch_idx:04d}  paths={len(batch)}  "
            f"catalog={len(rg_list)} RG + {len(mg_list)} MG ---"
        )
        sentinel = uuid.uuid4().hex[:12]
        prompt = make_assign_prompt(
            batch,
            node_attrs,
            rg_list,
            mg_list,
            sentinel,
            allow_new=True,
            allow_coherence=True,
        )
        print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)")
        json_part, dur, _, err = call_with_validation(
            prompt, sentinel, f"batch_{batch_idx:04d}", ASSIGN_MAX_TOKENS
        )
        if err or not json_part:
            print(f"  BATCH FAILED ({err}) — skipping")
            continue
        try:
            parsed = json.loads(json_part)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            continue

        rg_list, mg_list, resolved = resolve_assignments_and_update_catalog(
            rg_list, mg_list, parsed
        )
        atomic_write_json(
            active_path,
            {
                "n_risk_groups": len(rg_list),
                "n_mechanism_groups": len(mg_list),
                "risk_groups": rg_list,
                "mechanism_groups": mg_list,
            },
        )
        batch_out = batch_dir / f"batch_{batch_idx:04d}.json"
        atomic_write_json(
            batch_out,
            {
                "batch_idx": batch_idx,
                "n_input_paths": len(batch),
                "duration_sec": dur,
                "assignments": parsed.get("assignments", []),  # raw LLM output
                "resolved_assignments": resolved,  # canonical IDs
                "coherence_updates": parsed.get("coherence_updates", []),
            },
        )
        for upd in parsed.get("coherence_updates", []):
            upd["batch_idx"] = batch_idx
            coherence_log.append(upd)
        with open(coherence_log_path, "w", encoding="utf-8") as f:
            for upd in coherence_log:
                f.write(json.dumps(upd, ensure_ascii=False) + "\n")
        merged_path, n_rows = rebuild_merged_assignments_jsonl()
        print(
            f"  saved {batch_out.name}; catalog now {len(rg_list)} RG + {len(mg_list)} MG; "
            f"merged jsonl now {n_rows} rows"
        )
        # Regenerate xlsx so user can review in spreadsheet during the run.
        # If xlsx is open in Excel, write fails (PermissionError); log and continue —
        # the merged jsonl is still on disc and xlsx can be re-built standalone later.
        try:
            xlsx_result = build_xlsx_from_disc(
                paths_idx=paths_idx,
                node_attrs=node_attrs,
                edge_lookup=edge_lookup,
            )
            print(
                f"  xlsx refreshed: {xlsx_result['n_paths']} paths, "
                f"{xlsx_result['n_risk_groups']} RG + {xlsx_result['n_mechanism_groups']} MG"
            )
        except PermissionError as e:
            print(f"  xlsx WRITE BLOCKED (file open in Excel?): {e}")
            print(
                "  merged jsonl still updated; close Excel and re-run xlsx script standalone"
            )
        except Exception as e:
            print(f"  xlsx build FAILED ({type(e).__name__}): {e}")
            print("  this does NOT stop Pass B — merged jsonl is still on disc")


def _preserve_failed_partial(partial_path, batch_idx, reason):
    """Copy the partial stream to a permanently-named file so it isn't lost
    when the next batch resets partial_path. Used when JSON parse or stream
    fails, so the raw output can be inspected/recovered later.
    """
    import shutil

    if not partial_path.exists():
        return
    preserved = partial_path.with_suffix(f".batch_{batch_idx:04d}_failed.txt")
    try:
        shutil.copy2(partial_path, preserved)
        # Also write a small marker JSON alongside
        marker = partial_path.with_suffix(f".batch_{batch_idx:04d}_failed_meta.json")
        marker.write_text(
            json.dumps(
                {
                    "batch_idx": batch_idx,
                    "reason": reason,
                    "partial_file": str(preserved),
                    "size_bytes": preserved.stat().st_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"  preserved failed partial: {preserved.name} ({preserved.stat().st_size} bytes)",
            flush=True,
        )
    except Exception as e:
        print(f"  failed to preserve partial: {type(e).__name__}: {e}", flush=True)


def run_pass_b_sonnet(paths, node_attrs, n_batches=None, batch_size=75):
    """Sonnet-only Pass B routing (per forward_plan_2026_05_15_publication_path.md Phase 2a).

    - Catalog: read-only from phase2_doublet_seed_catalog.json (Opus seed)
    - Model: claude-sonnet-4-6
    - Schema: assignments-only; NO new groups; UNASSIGNED option enabled.
    - Per batch: atomic save to phase2_doublet_passb_batches/batch_NNNN.json
    - Tracks UNASSIGNED rows in phase2_doublet_passb_unassigned.jsonl for Opus REVIEW_C
    - State file phase2_doublet_passb_state.json holds batches-since-last-opus-review
      counter — caller checks this to know when to fire `--mode opus_review`.

    n_batches: max batches to run this invocation. Default = run until 10 batches
               since the last Opus review (i.e., one full checkpoint cycle), then stop
               so the user can launch `--mode opus_review` next.
    batch_size: paths per Sonnet call (default 75; Sonnet 200k context fits ~100 max).
    """
    from phase2_step4_phase2_doublet_to_xlsx import (
        build_xlsx_from_disc,
        build_edge_lookup,
        load_paths_indexed,
    )

    # Prefer active_catalog (post-Opus-REVIEW state) when it exists; fall back to seed
    # only when no REVIEW has run yet. Both files share the same {risk_groups, mechanism_groups}
    # schema. NEVER bootstrap from scratch; NEVER silently degrade.
    active_path = STEP1 / "phase2_doublet_active_catalog.json"
    seed_path = STEP1 / "phase2_doublet_seed_catalog.json"
    if active_path.exists():
        catalog_doc = json.loads(active_path.read_text(encoding="utf-8"))
        catalog_source = (
            f"active (post-review_idx={catalog_doc.get('review_idx', '?')})"
        )
    elif seed_path.exists():
        catalog_doc = json.loads(seed_path.read_text(encoding="utf-8"))
        catalog_source = "seed"
    else:
        raise FileNotFoundError(
            f"\nERROR: required artifact missing.\n"
            f"  Expected: {active_path} OR {seed_path}\n"
            f"  Produced by: `python phase2_step4_phase2_doublet_llm_grouping.py --mode seed`\n"
            f"  Pass B (Sonnet) requires either the seed catalog (first run) OR an active\n"
            f"  catalog from a prior Opus REVIEW pass. This script does NOT bootstrap from\n"
            f"  scratch and does NOT propose new groups (those are reserved for REVIEW_C).\n"
        )
    rg_list = list(catalog_doc["risk_groups"])
    mg_list = list(catalog_doc["mechanism_groups"])
    print(
        f"loaded fixed catalog: {len(rg_list)} RG + {len(mg_list)} MG "
        f"[source={catalog_source}]",
        flush=True,
    )

    passb_dir = STEP1 / "phase2_doublet_passb_batches"
    passb_dir.mkdir(parents=True, exist_ok=True)
    unassigned_log_path = STEP1 / "phase2_doublet_passb_unassigned.jsonl"
    state_path = STEP1 / "phase2_doublet_passb_state.json"
    partial_path = STEP1 / "phase2_doublet_passb_partial.txt"

    # Already-decided path_ids: seed assignments + completed passb batches +
    # legacy phase2_doublet_batches + REVIEW_C resolved paths from prior opus_review runs.
    # The seed catalog file (whether or not it's the catalog source we just loaded) always
    # carries the seed assignments — read it separately rather than from catalog_doc, since
    # active_catalog.json does NOT carry assignments.
    done_path_ids = set()
    if seed_path.exists():
        seed_assignments_doc = json.loads(seed_path.read_text(encoding="utf-8"))
        for a in seed_assignments_doc.get("assignments", []):
            if a.get("path_id"):
                done_path_ids.add(a["path_id"])
    for bf in sorted(passb_dir.glob("batch_*.json")):
        d = json.loads(bf.read_text(encoding="utf-8"))
        for a in d.get("resolved_assignments", []):
            if a.get("path_id"):
                done_path_ids.add(a["path_id"])
        for a in d.get("unassigned_rows", []):
            if a.get("path_id"):
                done_path_ids.add(a["path_id"])  # UNASSIGNED still counts as "decided"
    # Exclude legacy Opus pass_b batches if any
    legacy_dir = STEP1 / "phase2_doublet_batches"
    if legacy_dir.exists():
        for bf in sorted(legacy_dir.glob("batch_*.json")):
            d = json.loads(bf.read_text(encoding="utf-8"))
            for a in d.get("resolved_assignments", []):
                if a.get("path_id"):
                    done_path_ids.add(a["path_id"])
    # Exclude REVIEW_C-resolved paths (these were originally UNASSIGNED and have since been
    # routed by Opus REVIEW_C; do not re-route them via Sonnet).
    review_dir = STEP1 / "phase2_doublet_opus_reviews"
    if review_dir.exists():
        for rcf in sorted(review_dir.glob("review_c_*.json")):
            d = json.loads(rcf.read_text(encoding="utf-8"))
            for a in d.get("resolved_assignments", []):
                if a.get("path_id"):
                    done_path_ids.add(a["path_id"])
    print(
        f"already-decided paths (seed + passb + legacy + review_c): {len(done_path_ids)}",
        flush=True,
    )

    remaining = [p for p in paths if p["path_id"] not in done_path_ids]
    print(f"remaining paths to assign: {len(remaining)} of {len(paths)}", flush=True)
    if not remaining:
        print(
            "All paths assigned. Run --mode opus_review for final checkpoint, "
            "then Phase 4 meta-grouping.",
            flush=True,
        )
        return

    # Deterministic ordering across resumes
    rng = random.Random(BATCH_ORDER_RNG_SEED)
    rng.shuffle(remaining)

    # State: batches-since-last-opus-review counter
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "batches_since_review": 0,
            "total_passb_batches_run": 0,
            "last_review_at_batch_idx": -1,
        }

    # Determine next batch_idx
    existing_batches = sorted(passb_dir.glob("batch_*.json"))
    next_batch_idx = (
        max(
            [int(re.search(r"batch_(\d+)", f.name).group(1)) for f in existing_batches],
            default=-1,
        )
    ) + 1

    # Default: run until 10 since last review, then stop for user to run Opus review
    BATCHES_PER_CHECKPOINT = 10
    if n_batches is None:
        n_batches = BATCHES_PER_CHECKPOINT - state["batches_since_review"]
        n_batches = max(0, n_batches)
    n_remaining_batches_total = (len(remaining) + batch_size - 1) // batch_size
    n_batches = min(n_batches, n_remaining_batches_total)

    if n_batches == 0:
        print(
            f"State indicates {state['batches_since_review']} batches since last Opus review "
            f"already at {BATCHES_PER_CHECKPOINT} threshold.\n"
            f"Run `--mode opus_review` before more Sonnet batches.",
            flush=True,
        )
        return

    print(
        f"will run {n_batches} Sonnet batches (model=claude-sonnet-4-6) "
        f"starting at batch_{next_batch_idx:04d}",
        flush=True,
    )
    print(
        f"batches-since-last-opus-review will be: {state['batches_since_review']} -> "
        f"{state['batches_since_review'] + n_batches}",
        flush=True,
    )

    # Pre-build xlsx helpers ONCE per invocation
    print("loading edge data for xlsx builder ...", flush=True)
    with open(STEP1 / "graph_edge_data.pkl", "rb") as f:
        edge_data = pickle.load(f)
    edge_lookup = build_edge_lookup(edge_data)
    paths_idx = load_paths_indexed()
    print(
        f"  {len(edge_lookup)} EDGE-type edges; {len(paths_idx)} paths indexed",
        flush=True,
    )

    for bi in range(n_batches):
        batch_idx = next_batch_idx + bi
        batch = remaining[bi * batch_size : (bi + 1) * batch_size]
        print(
            f"\n--- passb_batch_{batch_idx:04d}  paths={len(batch)}  "
            f"catalog={len(rg_list)} RG + {len(mg_list)} MG (FIXED) ---",
            flush=True,
        )
        sentinel = uuid.uuid4().hex[:12]
        prompt = make_assign_prompt(
            batch,
            node_attrs,
            rg_list,
            mg_list,
            sentinel,
            allow_new=False,
            allow_coherence=False,
            allow_unassigned=True,
        )
        print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)

        json_part, dur, _, err = streaming_call_with_validation(
            prompt,
            sentinel,
            f"passb_batch_{batch_idx:04d}",
            partial_path,
            model="claude-sonnet-4-6",
        )
        if err or not json_part:
            print(
                f"  BATCH FAILED ({err}) — partial output at {partial_path.name}; "
                f"skipping this batch (will retry on next invocation)",
                flush=True,
            )
            # Preserve the failed partial with batch_idx so it isn't overwritten by
            # the next batch. The standard partial_path resets at next call's start.
            _preserve_failed_partial(
                partial_path, batch_idx, reason=f"stream_err={err}"
            )
            continue
        try:
            parsed = json.loads(json_part)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error at first attempt: {e}", flush=True)
            # Try recovery: Sonnet occasionally restarts its JSON output mid-stream
            # (observed batch_0001 + batch_0003 of 2026-05-15). The sentinel-check
            # passes because the response starts with `{` and ends with the marker,
            # but a stale partial prefix is concatenated with a complete restart.
            # Recovery: find the LAST occurrence of `{"assignments":[` and parse
            # from there. If that also fails, give up on this batch.
            recovery_marker = '{"assignments":['
            last_start = json_part.rfind(recovery_marker)
            recovered = None
            if last_start > 0:
                candidate = json_part[last_start:]
                try:
                    recovered = json.loads(candidate)
                    print(
                        f"  RECOVERED via rfind-restart: {len(recovered.get('assignments', []))} assignments "
                        f"(dropped {last_start} chars of stale prefix)",
                        flush=True,
                    )
                except json.JSONDecodeError as e2:
                    print(f"  RECOVERY ALSO FAILED: {e2}", flush=True)
            if recovered is None:
                _preserve_failed_partial(
                    partial_path, batch_idx, reason="json_parse_unrecoverable"
                )
                print(
                    f"  preserved partial at {partial_path.name}.batch_{batch_idx:04d}.txt for forensics",
                    flush=True,
                )
                continue
            parsed = recovered

        # Split assignments into (resolved, unassigned) based on per-row decision shape.
        # Sonnet schema: per path {"path_id", "risk_group": {"existing":"RG###"} OR {"unassigned":"..."},
        #                          "mechanism_group": {"existing":"MG###"} OR {"unassigned":"..."}}
        resolved_rows = []
        unassigned_rows = []
        for a in parsed.get("assignments", []):
            pid = a.get("path_id")
            if not pid:
                continue
            rg_field = a.get("risk_group", {})
            mg_field = a.get("mechanism_group", {})
            rg_id = rg_field.get("existing") if isinstance(rg_field, dict) else None
            mg_id = mg_field.get("existing") if isinstance(mg_field, dict) else None
            rg_unassigned = (
                rg_field.get("unassigned") if isinstance(rg_field, dict) else None
            )
            mg_unassigned = (
                mg_field.get("unassigned") if isinstance(mg_field, dict) else None
            )
            if rg_id and mg_id:
                resolved_rows.append(
                    {
                        "path_id": pid,
                        "risk_group_id": rg_id,
                        "mechanism_group_id": mg_id,
                    }
                )
            else:
                unassigned_rows.append(
                    {
                        "path_id": pid,
                        "risk_group_id": rg_id,
                        "mechanism_group_id": mg_id,
                        "risk_unassigned_reason": rg_unassigned,
                        "mechanism_unassigned_reason": mg_unassigned,
                        "batch_idx": batch_idx,
                    }
                )

        print(
            f"  -> {len(resolved_rows)} resolved + {len(unassigned_rows)} UNASSIGNED "
            f"(of {len(batch)} input)",
            flush=True,
        )

        batch_out = passb_dir / f"batch_{batch_idx:04d}.json"
        atomic_write_json(
            batch_out,
            {
                "batch_idx": batch_idx,
                "model": "claude-sonnet-4-6",
                "n_input_paths": len(batch),
                "duration_sec": dur,
                "assignments": parsed.get("assignments", []),
                "resolved_assignments": resolved_rows,
                "unassigned_rows": unassigned_rows,
            },
        )

        # Append UNASSIGNED rows to running log (Opus REVIEW_C input)
        if unassigned_rows:
            with open(unassigned_log_path, "a", encoding="utf-8") as f:
                for u in unassigned_rows:
                    f.write(json.dumps(u, ensure_ascii=False) + "\n")

        # Rebuild merged assignments jsonl (idempotent)
        merged_path, n_rows = rebuild_merged_assignments_jsonl()
        print(f"  saved {batch_out.name}; merged jsonl now {n_rows} rows", flush=True)

        # Auto-rebuild xlsx (graceful on Excel-open PermissionError)
        try:
            xlsx_result = build_xlsx_from_disc(
                paths_idx=paths_idx,
                node_attrs=node_attrs,
                edge_lookup=edge_lookup,
            )
            print(
                f"  xlsx refreshed: {xlsx_result['n_paths']} paths, "
                f"{xlsx_result['n_risk_groups']} RG + {xlsx_result['n_mechanism_groups']} MG",
                flush=True,
            )
        except PermissionError as e:
            print(f"  xlsx WRITE BLOCKED (Excel open?): {e}", flush=True)
            print(
                "  merged jsonl is still updated; close Excel and rebuild standalone",
                flush=True,
            )
        except Exception as e:
            print(f"  xlsx build FAILED ({type(e).__name__}): {e}", flush=True)
            print("  this does NOT stop Pass B — merged jsonl is on disc", flush=True)

        # Update state
        state["batches_since_review"] += 1
        state["total_passb_batches_run"] += 1
        atomic_write_json(state_path, state)

    print(flush=True)
    print("=== Pass B Sonnet checkpoint complete ===", flush=True)
    print(f"  ran {n_batches} batches this invocation", flush=True)
    print(
        f"  total passb batches so far: {state['total_passb_batches_run']}", flush=True
    )
    print(
        f"  batches since last Opus review: {state['batches_since_review']}", flush=True
    )
    if state["batches_since_review"] >= BATCHES_PER_CHECKPOINT:
        print(
            f"  >>> THRESHOLD REACHED ({state['batches_since_review']} >= {BATCHES_PER_CHECKPOINT}). "
            f"Run `--mode opus_review` next before more Sonnet batches.",
            flush=True,
        )


# ============================================================
# Opus REVIEW_A / B / C — catalog-refactor checkpoint (§19.13.2)
# ============================================================
# REVIEW_A: catalog-level audit. NO per-path content; sees group descriptions + per-group
#           assignment counts + UNASSIGNED-reason themes. Emits keep/rename/merge/deep_dive
#           per group. Renames + merges applied immediately; deep_dive groups passed to B.
# REVIEW_B: per-flagged-group deep dive. One Opus call per flagged group, sees up to 50
#           sample paths. Emits keep/rename/merge_with/split with per-subgroup path_ids.
# REVIEW_C: UNASSIGNED triage. Batched (REVIEW_C_BATCH_SIZE per call). Sees catalog
#           descriptions + UNASSIGNED path content + Sonnet reasons. Routes to existing
#           groups OR proposes new groups (>= MIN_GROUP_SIZE per new group).
# All three persist outputs to phase2_doublet_opus_reviews/review_{a,b,c}_NNN[_GID].json.
# All mutations go through group_remap.json + path_remap.json + active_catalog.json (with
# pre-write backups) so individual Sonnet batch files remain immutable.

REVIEW_A_MAX_TOKENS = 16384
REVIEW_B_MAX_TOKENS = 8192
REVIEW_C_MAX_TOKENS = 16384
REVIEW_C_BATCH_SIZE = 100
REVIEW_B_SAMPLE_SIZE = 50
REVIEW_A_THEMES_TOP_K = 20
REVIEW_RNG_SEED = 20260516
MAX_REVIEW_B_GROUPS = 15  # hard cap; REVIEW_A may flag more but only top-N (by current
# group size) are deep-dived this checkpoint. Token-budget safety:
# each REVIEW_B call costs ~20k input + ~1k output = ~3pp session.

# Shared paper-deliverable context — added 2026-05-17 (between REVIEW cycle 2 and 3).
# Methodological refinement, not a control variable: prior cycles 1+2 did not include
# this; cycles 3+ do. The change improves alignment between REVIEW decisions and the
# downstream paper deliverables described in Step4_Findings_Report.md §19.0 + §19.11.
PAPER_DELIVERABLE_CONTEXT = """\
THIS REVIEW SERVES A PUBLISHED-METHODOLOGY GOAL — keep these constraints in mind:

Corpus: Alignment Research Dataset (ARD, pre-2024 AI-safety papers). The doublet
catalog you are auditing/extending is intended to be a REPRESENTATIVE LOSSLESS
REDUCTION of ~8,954 paper-extracted intervention paths down to (a) a finite
risk-class vocabulary (RG###) and (b) a finite intervention-mechanism vocabulary
(MG###) — each path mapped to one (RG, MG) doublet.

Paper deliverables that depend on your decisions:
  (1) CATALOG-COVERAGE TABLE — mechanism classes × risk classes, showing which
      mechanisms address which risks at scale.
  (2) MANY-TO-FEW ANALYSIS — mechanism families that address MANY distinct risk
      classes (transferable mechanisms — the paper's headline finding).
  (3) GAP ANALYSIS — risk classes that are addressed by FEW mechanism classes
      (under-served risks; novel-intervention candidates).

What this means for your choices in this REVIEW pass:
  - Group NAMES must predict their contents — a reader interpreting (1)/(2)/(3)
    should be able to read meaning straight from the group_name + description.
  - Group GRANULARITY must support transferability detection — too-narrow
    mechanism groups hide cross-risk reuse; too-broad ones obscure mechanism
    distinctions. Favor groups that cleanly correspond to ONE causal pathway.
  - Risk groups should distinguish DIFFERENT failure modes, not different
    severities of the same failure mode. Two paths share a risk group iff
    they describe the SAME causal mechanism by which AI causes harm.
  - Mechanism groups should distinguish DIFFERENT causal mitigation pathways,
    not different surface-level implementations of the same pathway.
  - Singletons are forbidden for new groups (MIN_GROUP_SIZE=3) — only propose
    a new group if you expect >=3 paths corpus-wide will fit it.

PAPER GOAL — NOT just-another-clustering effort:
The contribution is NOT a field-navigation directory or cross-paper clustering
exercise. Those are free benefits. The actual goal is to capture the technical
detail of each intervention's MECHANISM (how it is proposed to reduce a specific
risk) at a level that supports two downstream uses:
  - reading off which mechanism families have evidence of working against which
    risks, and what stage of validation that evidence is at;
  - reasoning across the graph to propose NEW intervention candidates (mechanism
    transfer to under-served risks) that the field has not yet tried.

INTERVENTION vs RISK SEPARATION (load-bearing for paper analysis):
Risk-group and mechanism-group assignments must be made INDEPENDENTLY. Resist
the shortcut of inferring the risk from the mechanism (e.g., "interpretability
intervention => AI opacity risk"). Many interpretability mechanisms target
downstream harms (deception, misaligned power-seeking) where opacity is the
upstream enabler, not the named risk in the path. The risk node + first 1-2
body nodes are ground truth for risk-side assignment; the last 1-2 body nodes +
intervention are ground truth for mechanism-side assignment. The whole paper
analysis (mechanism-class x risk-class matrix; many-to-few transferability;
gap candidates) collapses if risk and mechanism are inferred from each other
rather than read independently from the path.
"""


def _passb_paths():
    return {
        "state": STEP1 / "phase2_doublet_passb_state.json",
        "unassigned": STEP1 / "phase2_doublet_passb_unassigned.jsonl",
        "passb_dir": STEP1 / "phase2_doublet_passb_batches",
        "active_catalog": STEP1 / "phase2_doublet_active_catalog.json",
        "merged_assignments": STEP1 / "phase2_doublet_assignments.jsonl",
        "seed": STEP1 / "phase2_doublet_seed_catalog.json",
        "review_dir": STEP1 / "phase2_doublet_opus_reviews",
        "partial": STEP1 / "phase2_doublet_opus_review_partial.txt",
        "group_remap": STEP1 / "phase2_doublet_group_remap.json",
        "path_remap": STEP1 / "phase2_doublet_path_remap.json",
    }


def _load_active_catalog_or_seed():
    P = _passb_paths()
    if P["active_catalog"].exists():
        d = json.loads(P["active_catalog"].read_text(encoding="utf-8"))
        return list(d["risk_groups"]), list(d["mechanism_groups"])
    if not P["seed"].exists():
        raise FileNotFoundError(
            f"\nERROR: required artifact missing.\n"
            f"  Expected: {P['seed']}\n"
            f"  Produced by: `python phase2_step4_phase2_doublet_llm_grouping.py --mode seed`\n"
            f"  Opus REVIEW requires a seed catalog. This script does NOT bootstrap a catalog\n"
            f"  from scratch and does NOT fall back to any other source.\n"
        )
    d = json.loads(P["seed"].read_text(encoding="utf-8"))
    return list(d["risk_groups"]), list(d["mechanism_groups"])


def _save_active_catalog(rg_list, mg_list, review_idx, kind_suffix):
    import shutil

    P = _passb_paths()
    out = P["active_catalog"]
    if out.exists():
        backup = out.with_suffix(f".pre_review_{kind_suffix}_{review_idx:03d}.json")
        try:
            shutil.copy2(out, backup)
        except Exception as e:
            print(f"  WARN: backup of {out.name} failed: {e}", flush=True)
    atomic_write_json(
        out,
        {
            "review_idx": review_idx,
            "applied_review_kind": kind_suffix,
            "n_risk_groups": len(rg_list),
            "n_mechanism_groups": len(mg_list),
            "risk_groups": rg_list,
            "mechanism_groups": mg_list,
        },
    )


def _load_passb_state():
    P = _passb_paths()
    if P["state"].exists():
        s = json.loads(P["state"].read_text(encoding="utf-8"))
        s.setdefault("total_reviews_run", 0)
        return s
    return {
        "batches_since_review": 0,
        "total_passb_batches_run": 0,
        "last_review_at_batch_idx": -1,
        "total_reviews_run": 0,
    }


def _save_passb_state(state):
    P = _passb_paths()
    atomic_write_json(P["state"], state)


def _load_unassigned_rows(skip_already_review_c_resolved=True):
    """Read Sonnet UNASSIGNED rows from phase2_doublet_passb_unassigned.jsonl.
    By default, filter out paths already resolved by an earlier REVIEW_C batch
    (so re-runs after a partial REVIEW_C failure only process the remaining
    unresolved paths, not the entire 545-row history).
    """
    P = _passb_paths()
    if not P["unassigned"].exists():
        return []
    rows = []
    for line in P["unassigned"].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if skip_already_review_c_resolved and P["review_dir"].exists():
        already_resolved = set()
        for rcf in sorted(P["review_dir"].glob("review_c_*.json")):
            d = json.loads(rcf.read_text(encoding="utf-8"))
            for a in d.get("resolved_assignments", []):
                if a.get("path_id"):
                    already_resolved.add(a["path_id"])
        before = len(rows)
        rows = [r for r in rows if r.get("path_id") not in already_resolved]
        if before != len(rows):
            print(
                f"  _load_unassigned_rows: filtered {before - len(rows)} "
                f"already-review_c-resolved; {len(rows)} remain",
                flush=True,
            )
    return rows


def _compute_group_stats(rg_list, mg_list):
    """Returns (rg_counts, mg_counts, doublet_counts) over the current merged assignments."""
    P = _passb_paths()
    rg_counts = {g["group_id"]: 0 for g in rg_list}
    mg_counts = {g["group_id"]: 0 for g in mg_list}
    doublet_counts = defaultdict(int)
    if not P["merged_assignments"].exists():
        return rg_counts, mg_counts, doublet_counts
    for line in P["merged_assignments"].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rg = r.get("risk_group_id")
        mg = r.get("mechanism_group_id")
        if rg in rg_counts:
            rg_counts[rg] += 1
        if mg in mg_counts:
            mg_counts[mg] += 1
        if rg and mg:
            doublet_counts[(rg, mg)] += 1
    return rg_counts, mg_counts, doublet_counts


def _sample_group_members(
    group_id, kind, paths_by_id, node_attrs, k=REVIEW_B_SAMPLE_SIZE
):
    """Random sample (seeded) of up to k assignment rows whose group on `kind` axis matches
    group_id. Each returned dict carries 'fmt_path_block' (rendered for the LLM) plus the
    original row fields."""
    P = _passb_paths()
    field = "risk_group_id" if kind == "risk" else "mechanism_group_id"
    matching = []
    if not P["merged_assignments"].exists():
        return matching
    for line in P["merged_assignments"].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get(field) == group_id:
            matching.append(r)
    rng = random.Random(REVIEW_RNG_SEED + hash(group_id) % 100000)
    rng.shuffle(matching)
    sampled = matching[:k]
    for r in sampled:
        path_rec = paths_by_id.get(r["path_id"])
        r["fmt_path_block"] = (
            fmt_path(path_rec, node_attrs)
            if path_rec
            else f"[{r['path_id']}] (NOT FOUND)"
        )
    return sampled


def _extract_unassigned_themes(unassigned_rows, k=REVIEW_A_THEMES_TOP_K):
    """Top-K UNASSIGNED reason themes by frequency, separated by axis."""
    from collections import Counter

    rg_reasons = Counter()
    mg_reasons = Counter()
    for r in unassigned_rows:
        rg_r = (r.get("risk_unassigned_reason") or "").strip()
        mg_r = (r.get("mechanism_unassigned_reason") or "").strip()
        if rg_r:
            rg_reasons[rg_r] += 1
        if mg_r:
            mg_reasons[mg_r] += 1
    themes = []
    for reason, n in rg_reasons.most_common(k):
        themes.append(f"(risk axis, seen {n}x) {reason[:240]}")
    for reason, n in mg_reasons.most_common(k):
        themes.append(f"(mechanism axis, seen {n}x) {reason[:240]}")
    return themes


# ---- Prompt makers ----------------------------------------------------------


def make_review_a_prompt(
    rg_list,
    mg_list,
    rg_counts,
    mg_counts,
    unassigned_themes,
    sentinel,
    allow_paper_context=True,
):
    """REVIEW_A: catalog-level audit. NO per-path content.
    allow_paper_context: if True (default), prepend PAPER_DELIVERABLE_CONTEXT to the prompt
    so Opus knows the ARD-corpus + downstream-deliverable framing. Set False only for
    sensitivity studies isolating the context effect."""
    ctx = PAPER_DELIVERABLE_CONTEXT + "\n\n" if allow_paper_context else ""
    rg_total = sum(rg_counts.values()) or 1
    mg_total = sum(mg_counts.values()) or 1
    rg_lines = []
    for g in rg_list:
        n = rg_counts.get(g["group_id"], 0)
        rg_lines.append(
            f"  {g['group_id']:<6} n={n:>4}  {g['group_name'][:62]:<62} | "
            f"{truncate(g['group_description'], 220)}"
        )
    mg_lines = []
    for g in mg_list:
        n = mg_counts.get(g["group_id"], 0)
        mg_lines.append(
            f"  {g['group_id']:<6} n={n:>4}  {g['group_name'][:62]:<62} | "
            f"{truncate(g['group_description'], 220)}"
        )
    themes_str = (
        "\n".join(f"  - {t}" for t in unassigned_themes) or "  (no themes extracted)"
    )

    return f"""You are auditing the doublet catalog for an AI-safety mechanism analysis.

{ctx}CONTEXT
- The catalog has TWO parallel taxonomies: RISK GROUPS (RG###) and MECHANISM GROUPS (MG###).
- A cheaper LLM ("Pass B routing") has assigned every resolved path to one RG and one MG.
  You do NOT see per-path content in this audit — only group descriptions + per-group
  assignment counts + a summary of cases the routing LLM flagged as UNASSIGNED.
- Your task: audit the catalog for fragmentation, redundancy, overly-broad groups, mis-named
  groups, and structural issues that group descriptions + counts can reveal.

DECISION CATEGORIES (output one per group):
  "keep"      — group is healthy as-is.
  "rename"    — same paths fit but the name/description is wrong or misleading. Provide
                new_name and new_description.
  "merge"     — group is redundant with another existing group. Provide target_group_id
                (the survivor). The merging group's paths will be reassigned to the
                survivor automatically. DO NOT pick a survivor with n=0; survivor must
                be a real, populated group.
  "deep_dive" — group needs per-path inspection before a decision (large count + suspected
                fragmentation, OR ambiguous scope). A separate REVIEW_B pass will look at
                sample paths from this group. Provide a 1-sentence deep_dive_reason.

GROUP HEALTH SIGNALS TO CONSIDER:
  - Very small groups (n=1 or n=2): merge candidate, OR a genuinely rare niche worth keeping.
  - Very large groups (>4-5% of total assignments): may be subsuming distinct mechanisms;
    consider deep_dive.
  - Two groups whose names/descriptions overlap strongly: merge one into the other.
  - Group name and description out of sync: rename.
  - Group description too vague to predict membership: rename or deep_dive.

DO NOT propose new groups here. UNASSIGNED triage happens in REVIEW_C — your job is to
clean up the EXISTING catalog. Do not modify any rationale around UNASSIGNED themes; they
are shown only as context for what REVIEW_C will later address.

============================================================
RISK GROUPS - {len(rg_list)} groups, total {rg_total} assignments
============================================================
{chr(10).join(rg_lines)}

============================================================
MECHANISM GROUPS - {len(mg_list)} groups, total {mg_total} assignments
============================================================
{chr(10).join(mg_lines)}

============================================================
SONNET UNASSIGNED-REASON THEMES (context only — do not act on these here)
============================================================
{themes_str}

============================================================
OUTPUT FORMAT (STRICT — validation will reject malformed responses)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences. No commentary.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.
- `rg_decisions` MUST contain exactly one entry per RG (every RG###, no skips, no dupes).
- `mg_decisions` MUST contain exactly one entry per MG (every MG###, no skips, no dupes).

Schema:

{{
  "rg_decisions": [
    {{"group_id": "RG001", "decision": "keep"}},
    {{"group_id": "RG002", "decision": "rename",
      "new_name": "...", "new_description": "..."}},
    {{"group_id": "RG003", "decision": "merge", "target_group_id": "RG017"}},
    {{"group_id": "RG004", "decision": "deep_dive",
      "deep_dive_reason": "<1 sentence>"}}
  ],
  "mg_decisions": [
    {{"group_id": "MG001", "decision": "keep"}}
  ],
  "audit_summary": "<2-4 sentences: structural health of catalog, % flagged for deep_dive, any cross-axis pattern>"
}}END_SENTINEL_{sentinel}

Produce the audit decisions now."""


def make_review_b_prompt(
    group, kind, members_with_paths, sentinel, min_group_size=MIN_GROUP_SIZE
):
    """REVIEW_B: deep-dive on ONE group. Shows the group + up to k sampled member paths."""
    kind_label = "RISK GROUP" if kind == "risk" else "MECHANISM GROUP"
    other_label = "MG (mechanism group)" if kind == "risk" else "RG (risk group)"
    other_field = "mechanism_group_id" if kind == "risk" else "risk_group_id"
    body_lines = []
    for r in members_with_paths:
        body_lines.append(
            f"--- path {r['path_id']}; currently {other_label}={r.get(other_field, '?')} ---"
        )
        body_lines.append(r.get("fmt_path_block", "(unavailable)"))
    body = "\n\n".join(body_lines)

    return f"""You are deep-diving on ONE {kind_label} flagged by the catalog-level audit.

{PAPER_DELIVERABLE_CONTEXT}

GROUP UNDER REVIEW
  group_id:    {group["group_id"]}
  group_name:  {group["group_name"]}
  description: {group["group_description"]}

This group currently holds the {len(members_with_paths)} sample path(s) below (randomly drawn
from its full membership). Read the path content carefully and decide:

DECISION CATEGORIES:
  "keep"       — sampled paths agree with the group description; no change.
  "rename"     — same paths but name/description is wrong. Provide new_name and new_description.
  "merge_with" — the sampled paths actually belong to another existing group. Provide
                 target_group_id (must be a different, existing group_id).
  "split"      — the sampled paths describe >= 2 distinct mechanisms that should be separated.
                 Provide a `subgroups` array; each subgroup has new_name, new_description,
                 and the path_ids from this sample that belong to it. EVERY sampled path_id
                 MUST appear in exactly one subgroup. Subgroups with < {min_group_size}
                 sampled paths will be merged back; only propose a split when each subgroup
                 has reasonable sampled population.

NEVER propose new groups here outside the `split.subgroups` mechanism. NEVER assign a path
to a group not in this sample's split subgroups.

============================================================
SAMPLE PATHS - {len(members_with_paths)} paths drawn from this group
============================================================
{body}

============================================================
OUTPUT FORMAT (STRICT — validation will reject malformed responses)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences. No commentary.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.

Schema:

{{
  "decision": "keep" | "rename" | "merge_with" | "split",
  "rationale": "<1-3 sentences on the evidence>",
  "new_name": "<rename only; else null>",
  "new_description": "<rename only; else null>",
  "target_group_id": "<merge_with only; else null>",
  "subgroups": [
    {{"new_name": "...", "new_description": "...", "path_ids": ["path_00042", "..."]}}
  ]
}}END_SENTINEL_{sentinel}

Produce the decision now."""


def make_review_c_prompt(
    unassigned_rows_with_paths,
    rg_list,
    mg_list,
    sentinel,
    min_group_size=MIN_GROUP_SIZE,
):
    """REVIEW_C: triage UNASSIGNED. Existing-catalog descriptions only + the unassigned paths."""
    body_lines = []
    for r in unassigned_rows_with_paths:
        reason_rg = r.get("risk_unassigned_reason") or ""
        reason_mg = r.get("mechanism_unassigned_reason") or ""
        cur_rg = r.get("risk_group_id") or "UNASSIGNED"
        cur_mg = r.get("mechanism_group_id") or "UNASSIGNED"
        body_lines.append(
            f"[{r['path_id']}] sonnet_current: RG={cur_rg} MG={cur_mg} | "
            f"risk_reason: {(reason_rg or '(n/a)')[:200]} | "
            f"mech_reason: {(reason_mg or '(n/a)')[:200]}"
        )
        body_lines.append(r.get("fmt_path_block", "(unavailable)"))
    body = "\n\n".join(body_lines)
    catalog = fmt_catalog_compact(rg_list, mg_list)
    n = len(unassigned_rows_with_paths)

    return f"""You are triaging paths the routing LLM (Sonnet) marked UNASSIGNED because no
existing RG or MG fit. Your job: assign each path to an existing group OR propose a NEW group
when a coherent cluster of >= {min_group_size} paths in THIS batch demands one.

{PAPER_DELIVERABLE_CONTEXT}

RULES
- Re-decide BOTH axes (risk_group, mechanism_group) for EVERY path. Do not assume Sonnet's
  partial decision was correct; it may have been UNASSIGNED on one axis only.
- Propose a NEW group ONLY when >= {min_group_size} of these {n} paths will fit it. Use the
  same proposed group_name VERBATIM across the paths that share it. Singleton new groups
  are forbidden — if only one path needs something genuinely new, force-fit it to the closest
  existing group and note the mismatch in the description-mismatch note (omit this field if
  the existing fit is fine).
- The `assignments` array MUST contain exactly {n} entries — one per input path.

============================================================
EXISTING CATALOG (descriptions only — existing-group paths NOT shown)
============================================================
{catalog}

============================================================
UNASSIGNED PATHS ({n} paths)
============================================================
{body}

============================================================
OUTPUT FORMAT (STRICT — validation will reject malformed responses)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences. No commentary.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.

Schema:

{{
  "assignments": [
    {{"path_id": "path_00042",
      "risk_group":      {{"existing": "RG017"}}  OR  {{"new": {{"group_name": "...", "group_description": "..."}}}},
      "mechanism_group": {{"existing": "MG023"}}  OR  {{"new": {{"group_name": "...", "group_description": "..."}}}}}}
  ]
}}END_SENTINEL_{sentinel}

Produce the assignments now."""


# ---- Apply-decision helpers --------------------------------------------------


def _apply_review_a_simple(group_list, decisions, kind):
    """Apply REVIEW_A 'keep', 'rename', 'merge' decisions. Returns
    (updated_group_list, flagged_for_deep_dive: list[group], merge_remap: dict)."""
    by_id = {g["group_id"]: g for g in group_list}
    flagged = []
    merge_remap = {}
    n_keep = n_rename = n_merge = n_deep = n_warn = 0
    for d in decisions:
        gid = d.get("group_id")
        if gid not in by_id:
            print(
                f"    WARN: REVIEW_A decision references unknown {kind} group_id={gid}; skipping",
                flush=True,
            )
            n_warn += 1
            continue
        action = d.get("decision")
        if action == "keep":
            n_keep += 1
        elif action == "rename":
            new_name = (d.get("new_name") or by_id[gid]["group_name"]).strip()
            new_desc = (
                d.get("new_description") or by_id[gid]["group_description"]
            ).strip()
            by_id[gid]["group_name"] = new_name
            by_id[gid]["group_description"] = new_desc
            n_rename += 1
        elif action == "merge":
            target = d.get("target_group_id")
            if target == gid or target not in by_id:
                print(
                    f"    WARN: REVIEW_A merge target {target} invalid for {gid}; skipping",
                    flush=True,
                )
                n_warn += 1
                continue
            merge_remap[gid] = target
            n_merge += 1
        elif action == "deep_dive":
            flagged.append(by_id[gid])
            n_deep += 1
        else:
            print(
                f"    WARN: unknown REVIEW_A decision '{action}' for {gid}; treating as keep",
                flush=True,
            )
            n_warn += 1
    new_list = [g for g in group_list if g["group_id"] not in merge_remap]
    print(
        f"    {kind}: keep={n_keep} rename={n_rename} merge={n_merge} deep_dive={n_deep} warn={n_warn}",
        flush=True,
    )
    return new_list, flagged, merge_remap


def _write_group_remap(rg_remap_new, mg_remap_new):
    """Merge new remap entries into phase2_doublet_group_remap.json (atomic, transitive)."""
    P = _passb_paths()
    if P["group_remap"].exists():
        cur = json.loads(P["group_remap"].read_text(encoding="utf-8"))
    else:
        cur = {"rg": {}, "mg": {}}
    cur.setdefault("rg", {}).update(rg_remap_new or {})
    cur.setdefault("mg", {}).update(mg_remap_new or {})
    cur["rg"] = _resolve_remap_transitive(cur["rg"])
    cur["mg"] = _resolve_remap_transitive(cur["mg"])
    atomic_write_json(P["group_remap"], cur)


def _write_path_remap(overrides):
    """Merge new per-path overrides into phase2_doublet_path_remap.json (atomic)."""
    P = _passb_paths()
    if P["path_remap"].exists():
        cur = json.loads(P["path_remap"].read_text(encoding="utf-8"))
    else:
        cur = {}
    for pid, override in (overrides or {}).items():
        cur.setdefault(pid, {}).update(override)
    atomic_write_json(P["path_remap"], cur)


def _next_group_id(group_list, prefix):
    """Return next sequential RG/MG id (e.g., 'RG042') above the current max."""
    max_idx = max(
        (
            int(g["group_id"][2:])
            for g in group_list
            if g["group_id"].startswith(prefix)
        ),
        default=0,
    )
    return f"{prefix}{max_idx + 1:03d}"


def _apply_review_b_decision(group, kind, decision, group_list):
    """Apply ONE REVIEW_B decision: keep/rename/merge_with/split. Returns
    (updated_group_list, merge_remap_addition: dict, path_remap_additions: dict).
    Splits append new groups to group_list and add per-path overrides."""
    gid = group["group_id"]
    by_id = {g["group_id"]: g for g in group_list}
    merge_add = {}
    path_overrides = {}
    action = decision.get("decision")
    if action == "keep":
        return group_list, merge_add, path_overrides
    if action == "rename":
        new_name = (decision.get("new_name") or group["group_name"]).strip()
        new_desc = (
            decision.get("new_description") or group["group_description"]
        ).strip()
        if gid in by_id:
            by_id[gid]["group_name"] = new_name
            by_id[gid]["group_description"] = new_desc
        return group_list, merge_add, path_overrides
    if action == "merge_with":
        target = decision.get("target_group_id")
        if target and target != gid and target in by_id:
            merge_add[gid] = target
            new_list = [g for g in group_list if g["group_id"] != gid]
            return new_list, merge_add, path_overrides
        print(
            f"    WARN: REVIEW_B merge_with target {target} invalid for {gid}; treating as keep",
            flush=True,
        )
        return group_list, merge_add, path_overrides
    if action == "split":
        subgroups = decision.get("subgroups") or []
        valid_subs = [
            s for s in subgroups if len(s.get("path_ids", [])) >= MIN_GROUP_SIZE
        ]
        if len(valid_subs) < 2:
            print(
                f"    WARN: REVIEW_B split for {gid} has < 2 valid subgroups "
                f"(min size {MIN_GROUP_SIZE}); treating as keep",
                flush=True,
            )
            return group_list, merge_add, path_overrides
        prefix = "RG" if kind == "risk" else "MG"
        new_list = list(group_list)
        field = "risk_group_id" if kind == "risk" else "mechanism_group_id"
        for sub in valid_subs:
            new_id = _next_group_id(new_list, prefix)
            new_list.append(
                {
                    "group_id": new_id,
                    "group_name": (sub.get("new_name") or "").strip(),
                    "group_description": (sub.get("new_description") or "").strip(),
                }
            )
            for pid in sub.get("path_ids", []):
                path_overrides.setdefault(pid, {})[field] = new_id
        new_list = [g for g in new_list if g["group_id"] != gid]
        merge_add[gid] = "SPLIT_REMOVED"
        return new_list, merge_add, path_overrides
    print(
        f"    WARN: unknown REVIEW_B decision '{action}' for {gid}; treating as keep",
        flush=True,
    )
    return group_list, merge_add, path_overrides


def _apply_review_c_assignments(parsed, rg_list, mg_list, batch_idx):
    """Apply REVIEW_C output: append new groups to catalog (idempotent by name), then return
    list of canonical-id resolved_assignments rows. Writes the per-batch review_c_NNN.json
    file under phase2_doublet_opus_reviews/."""
    P = _passb_paths()
    P["review_dir"].mkdir(parents=True, exist_ok=True)
    rg_list, mg_list, resolved = resolve_assignments_and_update_catalog(
        rg_list, mg_list, parsed
    )
    out_path = P["review_dir"] / f"review_c_{batch_idx:03d}.json"
    atomic_write_json(
        out_path,
        {
            "review_c_batch_idx": batch_idx,
            "n_input_paths": len(parsed.get("assignments", [])),
            "assignments": parsed.get("assignments", []),
            "resolved_assignments": resolved,
        },
    )
    return rg_list, mg_list, resolved, out_path


# ---- Runners ----------------------------------------------------------------


def run_opus_review_a(review_idx, partial_path):
    """REVIEW_A: catalog audit. Applies renames + merges; returns flagged-for-B groups."""
    P = _passb_paths()
    P["review_dir"].mkdir(parents=True, exist_ok=True)
    rg_list, mg_list = _load_active_catalog_or_seed()
    rg_counts, mg_counts, _ = _compute_group_stats(rg_list, mg_list)
    themes = _extract_unassigned_themes(
        _load_unassigned_rows(), k=REVIEW_A_THEMES_TOP_K
    )
    sentinel = uuid.uuid4().hex[:12]
    prompt = make_review_a_prompt(
        rg_list, mg_list, rg_counts, mg_counts, themes, sentinel
    )
    print(f"\n=== REVIEW_A (review_idx={review_idx:03d}) ===", flush=True)
    print(f"  catalog: {len(rg_list)} RG + {len(mg_list)} MG", flush=True)
    print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)
    json_part, dur, _, err = streaming_call_with_validation(
        prompt,
        sentinel,
        f"review_a_{review_idx:03d}",
        partial_path,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        _preserve_failed_partial(
            partial_path, review_idx, reason=f"review_a_stream_err={err}"
        )
        raise RuntimeError(f"REVIEW_A failed (err={err}); partial preserved")
    try:
        parsed = json.loads(json_part)
    except json.JSONDecodeError as e:
        print(f"  REVIEW_A JSON parse error: {e}", flush=True)
        recovery_marker = '{"rg_decisions":['
        last_start = json_part.rfind(recovery_marker)
        if last_start > 0:
            try:
                parsed = json.loads(json_part[last_start:])
                print(
                    f"  RECOVERED REVIEW_A via rfind-restart "
                    f"(dropped {last_start} chars)",
                    flush=True,
                )
            except json.JSONDecodeError as e2:
                _preserve_failed_partial(
                    partial_path, review_idx, reason="review_a_unrecoverable"
                )
                raise RuntimeError(f"REVIEW_A JSON unrecoverable: {e2}") from e
        else:
            _preserve_failed_partial(
                partial_path, review_idx, reason="review_a_unrecoverable"
            )
            raise RuntimeError(f"REVIEW_A JSON unrecoverable: {e}") from e

    out_path = P["review_dir"] / f"review_a_{review_idx:03d}.json"
    atomic_write_json(
        out_path,
        {
            "review_idx": review_idx,
            "duration_sec": dur,
            "raw_output": parsed,
            "catalog_size_pre": {"rg": len(rg_list), "mg": len(mg_list)},
        },
    )

    print("  applying REVIEW_A decisions ...", flush=True)
    rg_list_new, rg_flagged, rg_merge = _apply_review_a_simple(
        rg_list, parsed.get("rg_decisions", []), "risk"
    )
    mg_list_new, mg_flagged, mg_merge = _apply_review_a_simple(
        mg_list, parsed.get("mg_decisions", []), "mechanism"
    )
    _write_group_remap(rg_merge, mg_merge)
    _save_active_catalog(rg_list_new, mg_list_new, review_idx, kind_suffix="a")
    return (
        rg_list_new,
        mg_list_new,
        rg_flagged,
        mg_flagged,
        parsed.get("audit_summary", ""),
    )


def run_opus_review_b(
    rg_flagged,
    mg_flagged,
    review_idx,
    partial_path,
    paths_by_id,
    node_attrs,
    max_groups=MAX_REVIEW_B_GROUPS,
):
    """REVIEW_B: deep-dive on each flagged group. One Opus call per group.
    Caps the combined flagged set to `max_groups` (sorted by current group size desc) for
    token-budget safety. Excess flagged groups will be re-flagged in the next review cycle.
    """
    P = _passb_paths()
    rg_list, mg_list = _load_active_catalog_or_seed()
    rg_counts, mg_counts, _ = _compute_group_stats(rg_list, mg_list)
    total_flagged = len(rg_flagged) + len(mg_flagged)
    if total_flagged > max_groups:
        # Build combined ranked list (largest groups first), then cap.
        annotated = [
            (rg_counts.get(g["group_id"], 0), "risk", g) for g in rg_flagged
        ] + [(mg_counts.get(g["group_id"], 0), "mech", g) for g in mg_flagged]
        annotated.sort(key=lambda x: -x[0])
        kept = annotated[:max_groups]
        dropped = annotated[max_groups:]
        rg_flagged = [g for n, k, g in kept if k == "risk"]
        mg_flagged = [g for n, k, g in kept if k == "mech"]
        print(
            f"  REVIEW_B cap: {total_flagged} flagged > {max_groups} budget cap; "
            f"keeping top-{max_groups} by current size; "
            f"deferring {len(dropped)} to next review cycle",
            flush=True,
        )
    print(f"\n=== REVIEW_B (review_idx={review_idx:03d}) ===", flush=True)
    print(
        f"  {len(rg_flagged)} RG + {len(mg_flagged)} MG flagged for deep-dive "
        f"(capped at {max_groups})",
        flush=True,
    )

    aggregate_merge_rg = {}
    aggregate_merge_mg = {}
    aggregate_path_overrides = {}

    def do_one(group, kind):
        nonlocal rg_list, mg_list
        gid = group["group_id"]
        print(f"  --- {kind} {gid} '{group['group_name'][:60]}' ---", flush=True)
        members = _sample_group_members(gid, kind, paths_by_id, node_attrs)
        if len(members) < MIN_GROUP_SIZE:
            print(
                f"    only {len(members)} members; skipping deep-dive (too few to analyse)",
                flush=True,
            )
            return
        sentinel = uuid.uuid4().hex[:12]
        prompt = make_review_b_prompt(group, kind, members, sentinel)
        print(
            f"    prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens); "
            f"{len(members)} sample paths",
            flush=True,
        )
        json_part, dur, _, err = streaming_call_with_validation(
            prompt,
            sentinel,
            f"review_b_{review_idx:03d}_{gid}",
            partial_path,
            model="claude-opus-4-7",
        )
        if err or not json_part:
            _preserve_failed_partial(
                partial_path, review_idx, reason=f"review_b_{gid}_stream_err={err}"
            )
            print(
                f"    REVIEW_B for {gid} FAILED; partial preserved; skipping",
                flush=True,
            )
            return
        try:
            parsed = json.loads(json_part)
        except json.JSONDecodeError as e:
            print(f"    REVIEW_B JSON parse error for {gid}: {e}", flush=True)
            recovery_marker = '{"decision":'
            last_start = json_part.rfind(recovery_marker)
            if last_start > 0:
                try:
                    parsed = json.loads(json_part[last_start:])
                    print(
                        f"    RECOVERED REVIEW_B via rfind-restart "
                        f"(dropped {last_start} chars)",
                        flush=True,
                    )
                except json.JSONDecodeError as e2:
                    _preserve_failed_partial(
                        partial_path, review_idx, reason=f"review_b_{gid}_unrecoverable"
                    )
                    print(
                        f"    REVIEW_B for {gid} unrecoverable: {e2}; skipping",
                        flush=True,
                    )
                    return
            else:
                _preserve_failed_partial(
                    partial_path, review_idx, reason=f"review_b_{gid}_unrecoverable"
                )
                print(
                    f"    REVIEW_B for {gid} unrecoverable: {e}; skipping", flush=True
                )
                return

        out_path = P["review_dir"] / f"review_b_{review_idx:03d}_{gid}.json"
        atomic_write_json(
            out_path,
            {
                "review_idx": review_idx,
                "group_id": gid,
                "kind": kind,
                "duration_sec": dur,
                "n_members_sampled": len(members),
                "raw_output": parsed,
            },
        )

        if kind == "risk":
            new_list, merge_add, path_overrides = _apply_review_b_decision(
                group, kind, parsed, rg_list
            )
            rg_list = new_list
            for k, v in merge_add.items():
                if v != "SPLIT_REMOVED":
                    aggregate_merge_rg[k] = v
        else:
            new_list, merge_add, path_overrides = _apply_review_b_decision(
                group, kind, parsed, mg_list
            )
            mg_list = new_list
            for k, v in merge_add.items():
                if v != "SPLIT_REMOVED":
                    aggregate_merge_mg[k] = v
        for pid, override in path_overrides.items():
            aggregate_path_overrides.setdefault(pid, {}).update(override)
        print(
            f"    REVIEW_B {gid} decision={parsed.get('decision')}: applied", flush=True
        )

    for g in rg_flagged:
        do_one(g, "risk")
    for g in mg_flagged:
        do_one(g, "mechanism")

    if aggregate_merge_rg or aggregate_merge_mg:
        _write_group_remap(aggregate_merge_rg, aggregate_merge_mg)
    if aggregate_path_overrides:
        _write_path_remap(aggregate_path_overrides)
    _save_active_catalog(rg_list, mg_list, review_idx, kind_suffix="b")
    return rg_list, mg_list


def run_opus_review_c(
    review_idx, partial_path, paths_by_id, node_attrs, batch_size=REVIEW_C_BATCH_SIZE
):
    """REVIEW_C: UNASSIGNED triage in batches. May propose new groups (>= MIN_GROUP_SIZE)."""
    rg_list, mg_list = _load_active_catalog_or_seed()
    unassigned = _load_unassigned_rows()
    if not unassigned:
        print(f"\n=== REVIEW_C (review_idx={review_idx:03d}) ===", flush=True)
        print("  no UNASSIGNED rows on disc — skipping REVIEW_C", flush=True)
        return rg_list, mg_list, 0
    # Render each row's path block once
    for r in unassigned:
        path_rec = paths_by_id.get(r["path_id"])
        r["fmt_path_block"] = (
            fmt_path(path_rec, node_attrs)
            if path_rec
            else f"[{r['path_id']}] (NOT FOUND)"
        )

    print(f"\n=== REVIEW_C (review_idx={review_idx:03d}) ===", flush=True)
    print(
        f"  {len(unassigned)} UNASSIGNED rows; batch_size={batch_size}; "
        f"{(len(unassigned) + batch_size - 1) // batch_size} REVIEW_C calls",
        flush=True,
    )

    n_resolved_total = 0
    for bi, start in enumerate(range(0, len(unassigned), batch_size)):
        batch = unassigned[start : start + batch_size]
        batch_idx = review_idx * 100 + bi
        sentinel = uuid.uuid4().hex[:12]
        prompt = make_review_c_prompt(batch, rg_list, mg_list, sentinel)
        print(
            f"  --- review_c_{batch_idx:03d}  rows={len(batch)}  "
            f"catalog={len(rg_list)} RG + {len(mg_list)} MG ---",
            flush=True,
        )
        print(
            f"    prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True
        )
        json_part, dur, _, err = streaming_call_with_validation(
            prompt,
            sentinel,
            f"review_c_{batch_idx:03d}",
            partial_path,
            model="claude-opus-4-7",
        )
        if err or not json_part:
            _preserve_failed_partial(
                partial_path, batch_idx, reason=f"review_c_stream_err={err}"
            )
            print(
                f"    REVIEW_C batch {batch_idx} FAILED; partial preserved; skipping",
                flush=True,
            )
            continue
        try:
            parsed = json.loads(json_part)
        except json.JSONDecodeError as e:
            print(f"    REVIEW_C JSON parse error: {e}", flush=True)
            recovery_marker = '{"assignments":['
            last_start = json_part.rfind(recovery_marker)
            recovered = None
            if last_start > 0:
                try:
                    recovered = json.loads(json_part[last_start:])
                    print(
                        f"    RECOVERED REVIEW_C via rfind-restart "
                        f"(dropped {last_start} chars)",
                        flush=True,
                    )
                except json.JSONDecodeError as e2:
                    print(f"    REVIEW_C recovery also failed: {e2}", flush=True)
            if recovered is None:
                _preserve_failed_partial(
                    partial_path, batch_idx, reason="review_c_unrecoverable"
                )
                continue
            parsed = recovered

        rg_list, mg_list, resolved, out_path = _apply_review_c_assignments(
            parsed, rg_list, mg_list, batch_idx
        )
        n_resolved_total += len(resolved)
        print(
            f"    review_c_{batch_idx:03d}: {len(resolved)} resolved "
            f"(catalog now {len(rg_list)} RG + {len(mg_list)} MG)",
            flush=True,
        )

    _save_active_catalog(rg_list, mg_list, review_idx, kind_suffix="c")
    return rg_list, mg_list, n_resolved_total


def run_opus_review(paths, node_attrs, review_pass="all"):
    """Orchestrate Opus REVIEW_A -> REVIEW_B -> REVIEW_C. After all three, resets state's
    batches_since_review counter so Sonnet Pass B can resume.

    review_pass = 'a' | 'b' | 'c' | 'all'
      - 'a': only catalog audit (renames + merges); flag groups for later B
      - 'b': run B for groups flagged by the most recent A (re-reads its output JSON)
      - 'c': only UNASSIGNED triage
      - 'all': a then b then c, in one invocation
    """
    from phase2_step4_phase2_doublet_to_xlsx import (
        build_xlsx_from_disc,
        build_edge_lookup,
        load_paths_indexed,
    )

    P = _passb_paths()
    P["review_dir"].mkdir(parents=True, exist_ok=True)
    state = _load_passb_state()
    review_idx = state.get("total_reviews_run", 0) + 1
    paths_by_id = {p["path_id"]: p for p in paths}
    print("loading edge data for xlsx builder ...", flush=True)
    with open(STEP1 / "graph_edge_data.pkl", "rb") as f:
        edge_data = pickle.load(f)
    edge_lookup = build_edge_lookup(edge_data)
    paths_idx = load_paths_indexed()
    print(
        f"  {len(edge_lookup)} EDGE-type edges; {len(paths_idx)} paths indexed",
        flush=True,
    )

    rg_flagged_for_b = []
    mg_flagged_for_b = []
    audit_summary = ""

    def _build_pass_xlsx(kind_letter):
        """Write a snapshot xlsx after each review pass. Leaves the Sonnet-era xlsx
        (phase2_doublet_review_combined_v2.xlsx) untouched."""
        out_fp = (
            STEP1
            / f"phase2_doublet_review_after_review_{kind_letter}_{review_idx:03d}.xlsx"
        )
        try:
            xlsx_result = build_xlsx_from_disc(
                paths_idx=paths_idx,
                node_attrs=node_attrs,
                edge_lookup=edge_lookup,
                out_path=out_fp,
            )
            print(
                f"  xlsx (post-{kind_letter.upper()}) wrote: {out_fp.name} "
                f"({xlsx_result['n_paths']} paths, "
                f"{xlsx_result['n_risk_groups']} RG + {xlsx_result['n_mechanism_groups']} MG)",
                flush=True,
            )
        except PermissionError as e:
            print(f"  xlsx (post-{kind_letter.upper()}) WRITE BLOCKED: {e}", flush=True)
        except Exception as e:
            print(
                f"  xlsx (post-{kind_letter.upper()}) build FAILED: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

    if review_pass in ("a", "all"):
        _, _, rg_flagged_for_b, mg_flagged_for_b, audit_summary = run_opus_review_a(
            review_idx, P["partial"]
        )
        n_rows = rebuild_merged_assignments_jsonl()[1]
        print(f"  post-A merged jsonl: {n_rows} rows (group_remap applied)", flush=True)
        _build_pass_xlsx("a")

    if review_pass in ("b", "all"):
        if review_pass == "b" and not rg_flagged_for_b and not mg_flagged_for_b:
            # Re-read most recent A output to find flagged groups
            a_files = sorted(P["review_dir"].glob("review_a_*.json"))
            if a_files:
                latest = json.loads(a_files[-1].read_text(encoding="utf-8"))
                raw = latest.get("raw_output", {})
                rg_list_cur, mg_list_cur = _load_active_catalog_or_seed()
                rg_by_id = {g["group_id"]: g for g in rg_list_cur}
                mg_by_id = {g["group_id"]: g for g in mg_list_cur}
                rg_flagged_for_b = [
                    rg_by_id[d["group_id"]]
                    for d in raw.get("rg_decisions", [])
                    if d.get("decision") == "deep_dive"
                    and d.get("group_id") in rg_by_id
                ]
                mg_flagged_for_b = [
                    mg_by_id[d["group_id"]]
                    for d in raw.get("mg_decisions", [])
                    if d.get("decision") == "deep_dive"
                    and d.get("group_id") in mg_by_id
                ]
                print(
                    f"  loaded {len(rg_flagged_for_b)} RG + {len(mg_flagged_for_b)} MG "
                    f"flagged groups from {a_files[-1].name}",
                    flush=True,
                )
            else:
                print(
                    "  no prior review_a_*.json on disc; nothing to deep-dive",
                    flush=True,
                )
        run_opus_review_b(
            rg_flagged_for_b,
            mg_flagged_for_b,
            review_idx,
            P["partial"],
            paths_by_id,
            node_attrs,
        )
        n_rows = rebuild_merged_assignments_jsonl()[1]
        print(
            f"  post-B merged jsonl: {n_rows} rows (group_remap + path_remap applied)",
            flush=True,
        )
        _build_pass_xlsx("b")

    n_review_c_resolved = 0
    if review_pass in ("c", "all"):
        _, _, n_review_c_resolved = run_opus_review_c(
            review_idx, P["partial"], paths_by_id, node_attrs
        )
        n_rows = rebuild_merged_assignments_jsonl()[1]
        print(
            f"  post-C merged jsonl: {n_rows} rows (REVIEW_C batches included)",
            flush=True,
        )
        _build_pass_xlsx("c")

    # Reset state counter only when 'all' completed; partial passes leave counter as-is
    if review_pass == "all":
        state["total_reviews_run"] = review_idx
        state["batches_since_review"] = 0
        state["last_review_at_batch_idx"] = state.get("total_passb_batches_run", 0)
        _save_passb_state(state)
        print("\n=== Opus REVIEW (A+B+C) complete ===", flush=True)
        print(f"  audit_summary (REVIEW_A): {audit_summary}", flush=True)
        print(
            f"  REVIEW_C resolved {n_review_c_resolved} previously-UNASSIGNED paths",
            flush=True,
        )
        print(
            "  state.batches_since_review reset to 0; Sonnet Pass B can resume.",
            flush=True,
        )
    else:
        print(f"\n=== Opus REVIEW (pass={review_pass}) complete ===", flush=True)
        print(
            "  state counter NOT reset (partial pass). Run --review-pass all next or "
            "complete remaining passes individually before resuming Sonnet.",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["seed", "review", "smoke", "full", "pass_b_sonnet", "opus_review"],
        required=True,
    )
    ap.add_argument(
        "--n-paths",
        type=int,
        default=None,
        help="Override SEED_SAMPLE_SIZE for seed mode (smoke testing). "
        "When < SEED_SAMPLE_SIZE, output is tagged (SMOKE) but written to "
        "canonical paths; archive the result before re-running with default.",
    )
    ap.add_argument(
        "--n-batches",
        type=int,
        default=None,
        help="For pass_b_sonnet: max batches to run this invocation. "
        "Default = run until 10 batches since last Opus review, then stop.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=75,
        help="For pass_b_sonnet: paths per Sonnet call (default 75).",
    )
    ap.add_argument(
        "--review-pass",
        choices=["a", "b", "c", "all"],
        default="all",
        help="For opus_review: which sub-pass to run "
        "(a=catalog audit, b=per-group deep dive, c=UNASSIGNED triage, "
        "all=A then B then C). Default 'all'.",
    )
    args = ap.parse_args()

    paths, node_attrs = load_paths_and_attrs()

    if args.mode == "seed":
        run_seed(paths, node_attrs, n_paths=args.n_paths)
    elif args.mode == "review":
        run_review(paths, node_attrs)
    elif args.mode == "smoke":
        run_smoke(paths, node_attrs)
    elif args.mode == "full":
        run_full(paths, node_attrs)
    elif args.mode == "pass_b_sonnet":
        run_pass_b_sonnet(
            paths, node_attrs, n_batches=args.n_batches, batch_size=args.batch_size
        )
    elif args.mode == "opus_review":
        run_opus_review(paths, node_attrs, review_pass=args.review_pass)


if __name__ == "__main__":
    main()
