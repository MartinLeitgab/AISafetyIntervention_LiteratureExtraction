"""pilot_100_path_axis_discovery.py — 100-path Opus pilot for axes + classes design.

Per restart_plan_2026_05_17.md §4 (revised): a single Opus call sees 100 paths
(1 per source paper, sampled uniformly from the deduped 3,356-path corpus) and
is asked to:
  (1) Propose 3-9 orthogonal navigation axes for classifying the paths.
      Each axis has a controlled vocabulary; values must NOT duplicate
      what's captured in harm_class or mechanism_class.
  (2) Build an initial harm_class taxonomy (no meta-level catch-alls;
      capability-gap-only RGs get the explicit harm_target=capability-gap tag
      instead of being misfiled as peer human-risks).
  (3) Build an initial mechanism_class taxonomy (general transferable
      mechanism families; not one-paper interventions).
  (4) For each of the 100 paths, emit: (harm_class_id, mech_class_id,
      axis_values_per_axis, confidence) with continuum-aware assignment.
  (5) Critique the architecture — overlap between classes and axes,
      missing axes, granularity concerns.

Output: phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/
         phase2_pilot_100_axis_discovery.json

Class A. Estimated cost: ~80-100pp Opus (single call), ~10-15 min wall.
Streams to disc via existing streaming_call_with_validation wrapper.
"""

from __future__ import annotations
import json
import sys
import uuid
import random
import pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M

DEDUPED_PATHS = M.ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
PILOT_N = 100  # bumped back after 50-path pilot cost ~10pp actual (vs 50pp est)
OUT_FP = M.STEP1 / f"phase2_pilot_{PILOT_N}paths_axis_discovery.json"
PARTIAL_FP = M.STEP1 / f"phase2_pilot_{PILOT_N}paths_axis_discovery_partial.txt"
PILOT_RNG_SEED = 20260517  # distinct from prior seeds for clean lineage


def load_deduped_paths():
    print(f"loading deduped paths from {DEDUPED_PATHS.name} ...", flush=True)
    paths = []
    with open(DEDUPED_PATHS, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                d = json.loads(line)
                d["path_id"] = f"path_{i:05d}_dedup"  # distinct namespace
                paths.append(d)
    print(f"  {len(paths)} deduped paths", flush=True)
    return paths


def sample_one_per_paper(paths, node_attrs, target_n, seed):
    """Sample target_n paths with at most 1 per source URL."""
    by_url = defaultdict(list)
    for p in paths:
        urls = set()
        for nid in p.get("path", []):
            attrs = node_attrs.get(int(nid)) or node_attrs.get(nid) or {}
            url = attrs.get("url")
            if url:
                urls.add(url)
        if len(urls) == 1:
            by_url[next(iter(urls))].append(p)
    print(f"  unique source papers: {len(by_url)}", flush=True)
    rng = random.Random(seed)
    # Pick one path per paper, then sample target_n from the deduped set
    one_per_paper = [rng.choice(plist) for plist in by_url.values()]
    rng.shuffle(one_per_paper)
    return one_per_paper[:target_n]


def build_pilot_prompt(sample, node_attrs, sentinel):
    body = "\n\n".join(M.fmt_path(p, node_attrs) for p in sample)
    n = len(sample)
    return f"""You are designing the canonical classification architecture for a paper that
analyses {n} sample paths drawn 1-per-source-paper from the Alignment Research
Dataset (ARD) corpus. The final analysis will scale this to ~3,356 deduped paths.

{M.PAPER_DELIVERABLE_CONTEXT}

============================================================
PATH ROLES CONTEXT
============================================================

{M.SUBTYPE_DEFINITIONS}

============================================================
YOUR TASK — propose architecture + apply to these 100 paths
============================================================

Step 1 — PROPOSE 3-9 ORTHOGONAL NAVIGATION AXES.
Each axis describes one independent dimension of variation across the corpus
that a researcher might want to slice on. Each axis has a CONTROLLED VOCABULARY
(closed enum of possible values). Axes must NOT duplicate information captured
in the harm_class or mechanism_class taxonomies (Steps 2-3 below). Axes should
ALSO emit a per-path tag; risk-side axes (e.g., severity) tag the path's
risk node, intervention-side axes (e.g., lifecycle_stage) tag the intervention.

EXAMPLES (use as inspiration; propose your own set based on what these 100
paths reveal):
  - lifecycle_stage  (pre-train / fine-tune / inference / deployment / governance / monitoring / RL-RLHF / ...)
  - modality         (vision / LLM / RL-agent / embodied / multimodal / general / ...)
  - methodology      (algorithmic / data-centric / architectural / process-policy / human-in-loop / ...)
  - severity         (catastrophic / serious / moderate / minor)        [tags the RISK node]
  - reversibility    (recoverable / irreversible / catastrophic)        [tags the RISK node]
  - emergence_stage  (training / deployment / scaling / post-deployment) [tags the RISK node]
  - harm_target      (human-life / human-flourishing / economic / institutional / capability-gap-only)

DO NOT include stakeholder (redundant with methodology) or validation_maturity
(quality cuts filter for maturity >=3, no variance). DO NOT include cost-class
(too speculative to tag consistently).

ALLOW "OTHER" + free-text per axis when no enum value fits (LLM may extend
the controlled vocabulary in later runs).

Step 2 — BUILD INITIAL harm_class TAXONOMY.
Target ~10-25 harm classes across these 100 paths. Each is a discrete causal
failure mode by which AI causes harm to humans. Apply continuum awareness:
when the risk node is meta-level (e.g., "existential catastrophe"), use the
first 1-2 body nodes to identify the specific causal mechanism and prefer the
object-level class over a meta catch-all. NO meta-level catch-alls. Paths
where the entire chain is about ML engineering / capability gaps (e.g.,
intractable inference, sample inefficiency) get a CAPABILITY_GAP-flagged
harm_class — keep them separate from human-risk classes.

Step 3 — BUILD INITIAL mechanism_class TAXONOMY.
Target ~25-45 mechanism classes across these 100 paths. Each is a transferable
intervention pattern. Apply continuum awareness: when the intervention node is
a specific program/acronym/one-shot (e.g., "Run AGISF reading group"), use
the body to infer the GENERAL transferable mechanism class (e.g., "structured
peer-learning curriculum for AI safety education"). Prefer mechanism families
that recur across the corpus over single-paper specifics. MIN_GROUP_SIZE = 3
for new groups (only propose if you expect >=3 of these 100 paths fit it).

Step 4 — ASSIGN ALL 100 PATHS.
Each path gets:
  - harm_class_id (HC###)
  - mechanism_class_id (MC###)
  - One value per proposed axis (or "OTHER: <free_text>" if controlled vocab
    doesn't fit)
  - confidence (1-5, your judgment on assignment fit)
EVERY path must be assigned to every axis; no missing values.

Step 5 — ARCHITECTURE CRITIQUE.
Last section of output. Address: any overlap between axes and classes you
detected; missing axes you couldn't propose given only 100 paths but expect
will be needed at corpus scale; any harm/mech classes you flagged as
borderline (low confidence on definition); how your proposed architecture
serves the paper's three deliverables (faithfulness, scale catalog,
novel-intervention candidates).

============================================================
INPUT PATHS ({n} paths)
============================================================

{body}

============================================================
OUTPUT FORMAT (STRICT — validation will reject malformed responses)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences. No commentary.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.
- `assignments` array MUST contain exactly {n} entries.

Schema:

{{
  "axes": [
    {{"axis_name": "lifecycle_stage",
      "axis_kind": "intervention" | "risk",
      "values": ["pre-train", "fine-tune", "...", "OTHER"],
      "rationale": "<1 sentence>"}},
    ...
  ],
  "harm_classes": [
    {{"class_id": "HC001",
      "class_name": "<short distinctive name>",
      "class_description": "<1-2 sentences naming the causal failure mode>",
      "is_capability_gap": true | false}},
    ...
  ],
  "mechanism_classes": [
    {{"class_id": "MC001",
      "class_name": "<short distinctive name>",
      "class_description": "<1-2 sentences naming the transferable mechanism>"}},
    ...
  ],
  "assignments": [
    {{"path_id": "path_NNNNN_dedup",
      "harm_class_id": "HC###",
      "mechanism_class_id": "MC###",
      "axis_values": {{"lifecycle_stage": "fine-tune", "modality": "LLM", ...}},
      "confidence": 4}},
    ... (one entry per input path; no path missing; no duplicates)
  ],
  "architecture_critique": "<3-6 sentences: axis-class overlap, missing axes, low-confidence classes, fit to paper deliverables>"
}}END_SENTINEL_{sentinel}

Produce the architecture and assignments now."""


def main():
    if OUT_FP.exists():
        print(
            f"[idempotent skip] {OUT_FP.name} already exists. Delete to re-run.",
            flush=True,
        )
        return
    paths = load_deduped_paths()
    print("loading node_attrs ...", flush=True)
    with open(M.STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    print(f"  {len(na)} nodes", flush=True)

    sample = sample_one_per_paper(paths, na, PILOT_N, PILOT_RNG_SEED)
    print(f"sampled {len(sample)} paths (1 per paper, target {PILOT_N})", flush=True)

    sentinel = uuid.uuid4().hex[:12]
    prompt = build_pilot_prompt(sample, na, sentinel)
    print(f"prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)

    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        "pilot_100_axis",
        PARTIAL_FP,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(
            f"\nPILOT FAILED ({err}). Partial preserved at {PARTIAL_FP.name}.",
            flush=True,
        )
        sys.exit(2)
    try:
        parsed = json.loads(json_part)
    except json.JSONDecodeError as e:
        print(f"\nJSON parse error at first attempt: {e}", flush=True)
        last = json_part.rfind('{"axes":')
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

    M.atomic_write_json(
        OUT_FP,
        {
            "pilot_n": PILOT_N,
            "duration_sec": dur,
            "n_axes": len(parsed.get("axes", [])),
            "n_harm_classes": len(parsed.get("harm_classes", [])),
            "n_mech_classes": len(parsed.get("mechanism_classes", [])),
            "n_assignments": len(parsed.get("assignments", [])),
            "raw_output": parsed,
            "input_path_ids": [p["path_id"] for p in sample],
        },
    )

    print(f"\nwrote {OUT_FP}", flush=True)
    print(f"  axes proposed:          {len(parsed.get('axes', []))}", flush=True)
    print(
        f"  harm_classes proposed:  {len(parsed.get('harm_classes', []))}", flush=True
    )
    print(
        f"  mech_classes proposed:  {len(parsed.get('mechanism_classes', []))}",
        flush=True,
    )
    print(f"  assignments:            {len(parsed.get('assignments', []))}", flush=True)
    if parsed.get("architecture_critique"):
        print(
            f"\narchitecture_critique:\n{parsed['architecture_critique']}", flush=True
        )


if __name__ == "__main__":
    main()
