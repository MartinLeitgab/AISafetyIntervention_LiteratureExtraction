"""pilot_100_path_discovery_v2.py — Discovery-from-scratch v2 of the 100-path
pilot, applying architecture changes:
  - fit_score (independent from confidence)
  - unassigned option for harm_class and mechanism_class
  - paper-goal framing (NOT just-another-clustering)
  - risk/intervention decoupling instruction
  - MIN_GROUP_SIZE=3 explicit in output discipline (drops singletons)
  - harm_target_evidence per assignment

Uses the SAME 100 input path_ids as v1 (read from v1's input_path_ids list).
Writes to a separate namespace; v1 outputs are untouched.

Outputs:
  phase2_pilot_v2_100paths_discovery.json       - raw + parsed Opus output
  phase2_pilot_v2_100paths_discovery_partial.txt - streaming partial

Class A. Estimated ~12-14pp Opus. Wall-clock ~7-8 min.
"""

from __future__ import annotations
import json
import sys
import uuid
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M

PILOT_V1_FP = M.STEP1 / "phase2_pilot_100paths_axis_discovery.json"
DEDUPED_PATHS = M.ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
OUT_FP = M.STEP1 / "phase2_pilot_v2_100paths_discovery.json"
PARTIAL_FP = M.STEP1 / "phase2_pilot_v2_100paths_discovery_partial.txt"


def load_deduped_paths():
    print(f"loading deduped paths from {DEDUPED_PATHS.name} ...", flush=True)
    paths = []
    with open(DEDUPED_PATHS, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                d = json.loads(line)
                d["path_id"] = f"path_{i:05d}_dedup"
                paths.append(d)
    print(f"  {len(paths)} deduped paths", flush=True)
    return paths


def build_pilot_v2_prompt(sample, node_attrs, sentinel):
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
YOUR TASK — propose architecture + apply to these {n} paths
============================================================

Step 1 — PROPOSE 3-9 ORTHOGONAL NAVIGATION AXES.
Each axis is one independent dimension of variation across the corpus, with
a CONTROLLED VOCABULARY (closed enum). Axes must NOT duplicate information
captured in harm_class or mechanism_class (Steps 2-3). Each axis has axis_kind:
  - "intervention" — tags the path's intervention node
  - "risk"         — tags the path's risk node
Allow "OTHER:<free_text>" per axis when controlled vocab doesn't fit.

EXAMPLES (use as inspiration; design what these {n} paths reveal):
  - lifecycle_stage   (intervention) — pre-train, fine-tune, RL-training, model-design,
                                          pre-deployment-testing, deployment-runtime,
                                          post-deployment-audit, governance-policy, ...
  - modality          (intervention) — LLM, RL-agent, vision, multi-agent, robotics, ...
                                          (do NOT propose NLP as separate from LLM)
  - methodology       (intervention) — algorithmic, architectural, data-centric, ...
  - severity          (risk)         — catastrophic-existential, serious, moderate, minor
  - emergence_stage   (risk)         — training-time, deployment-runtime, post-deployment,
                                          scaling, structural-societal
  - harm_target       (risk)         — human-survival, human-flourishing-rights, economic,
                                          institutional-governance, scientific-truth,
                                          environmental, capability-gap-only

DO NOT include stakeholder (redundant with methodology) or validation_maturity
(quality cuts filter for maturity>=3, no variance). DO NOT include cost-class
(too speculative).

Step 2 — BUILD INITIAL harm_class TAXONOMY.
Target ~15-25 harm classes. Each = a discrete causal failure mode by which AI
causes harm to humans. Apply CONTINUUM AWARENESS: when the risk node is
meta-level (e.g., "existential catastrophe"), use the first 1-2 body nodes
to identify the specific causal mechanism and prefer the OBJECT-LEVEL class.
NO meta-level catch-alls.

Capability-gap framing: a path whose risk chain describes a PURE ML
capability gap (e.g., sample-inefficiency, training-instability, semantic
representation limits) gets a class with `is_capability_gap=true`. The
capability-gap tag describes the harm-chain side ONLY — it does NOT claim the
intervention has no safety value. Interventions tagged here may still be
safety-load-bearing via downstream chains.

MIN_GROUP_SIZE = 3 — only propose a class you expect >=3 of THESE {n} paths
will fit. Singletons forbidden; if a path has no >=3-peer class, use the
"unassigned" mechanism (Step 4) instead of inventing a singleton.

Step 3 — BUILD INITIAL mechanism_class TAXONOMY.
Target ~25-45 mechanism classes. Each = a transferable intervention pattern.
CONTINUUM AWARENESS: when the intervention node is a specific program/acronym
(e.g., "Run AGISF reading group"), use the body to infer the GENERAL
transferable mechanism family (e.g., "structured peer-learning curriculum for
AI safety education"). Prefer cross-paper-recurring mechanism families over
single-paper specifics. MIN_GROUP_SIZE = 3.

Step 4 — ASSIGN ALL {n} PATHS.
Each path gets:
  - harm_class_id (HC###) OR {{"unassigned": true, "reason": "..."}}
  - mechanism_class_id (MC###) OR {{"unassigned": true, "reason": "..."}}
  - One value per proposed axis (or "OTHER:<free_text>")
  - confidence  (1-5): how CLEARLY YOU READ the path's causal chain
                       (epistemic clarity, independent of fit_score)
  - fit_score   (1-5): how WELL the assigned harm + mechanism classes
                       ACTUALLY FIT the path
  - fit_note    (string, required if fit_score<=3 or unassigned): 1-clause justification
  - harm_target_evidence (string): 1-clause naming which risk-side node or
                                      description supports the harm_target axis value

UNASSIGNED DISCIPLINE:
- Use "unassigned" when no class with >=3 peers fits well. Bad force-fits
  contaminate the cross-paper matrix; unassigned paths are honest gaps that
  consolidation handles later.
- harm_class and mechanism_class are INDEPENDENT — a path may have one assigned
  and the other unassigned.

RISK / INTERVENTION DECOUPLING (REQUIRED — paper analysis depends on this):
Read risk-class and mechanism-class INDEPENDENTLY. Resist inferring risk from
mechanism (e.g., "interpretability intervention => AI opacity risk"). Many
interpretability mechanisms address downstream harms (deception, power-seeking)
where opacity is upstream enabler, not the named risk. The risk node + first
1-2 body nodes are ground truth for risk-side; last 1-2 body + intervention
are ground truth for mechanism-side. The paper's mechanism x risk matrix and
many-to-few transferability collapse if risk and mechanism are inferred from
each other rather than read independently from the path.

Step 5 — ARCHITECTURE CRITIQUE.
Address: axis-class overlap; missing axes; low-confidence/borderline classes;
classes you considered merging vs splitting; how the architecture serves the
paper's 3 deliverables (faithfulness, scale catalog, novel-intervention
candidates from cross-mechanism transfer).

============================================================
INPUT PATHS ({n} paths)
============================================================

{body}

============================================================
OUTPUT FORMAT (STRICT — validation will reject malformed responses)
============================================================

- Output ONLY one JSON object. No preamble. No markdown fences. No commentary.
- Start with `{{`. After closing `}}` append literal sentinel `END_SENTINEL_{sentinel}` on the same line.
- `assignments` array MUST contain exactly {n} entries (one per input path; no missing; no duplicates).

Schema:

{{
  "axes": [
    {{"axis_name": "lifecycle_stage",
      "axis_kind": "intervention" | "risk",
      "values": ["pre-train", "fine-tune", "...", "OTHER"],
      "rationale": "<1 sentence>"}}
  ],
  "harm_classes": [
    {{"class_id": "HC001",
      "class_name": "<short distinctive name>",
      "class_description": "<1-2 sentences naming the causal failure mode>",
      "is_capability_gap": true | false,
      "expected_n_paths": <int, your estimate of how many of the {n} fit>}}
  ],
  "mechanism_classes": [
    {{"class_id": "MC001",
      "class_name": "<short distinctive name>",
      "class_description": "<1-2 sentences naming the transferable mechanism>",
      "expected_n_paths": <int>}}
  ],
  "assignments": [
    {{"path_id": "path_NNNNN_dedup",
      "harm_class_id": "HC###"  OR  {{"unassigned": true, "reason": "..."}},
      "mechanism_class_id": "MC###"  OR  {{"unassigned": true, "reason": "..."}},
      "axis_values": {{"lifecycle_stage": "fine-tune", "modality": "LLM", "...": "..."}},
      "harm_target_evidence": "<1 clause citing risk-side support for harm_target value>",
      "confidence": 4,
      "fit_score": 4,
      "fit_note": "<required when fit_score<=3 OR unassigned>"}}
  ],
  "architecture_critique": "<3-6 sentences: overlap, missing axes, borderline classes, merge/split considerations, fit to paper deliverables>"
}}END_SENTINEL_{sentinel}

Produce the architecture and assignments now."""


def main():
    if OUT_FP.exists():
        print(f"[idempotent skip] {OUT_FP.name} exists. Delete to re-run.", flush=True)
        return
    if not PILOT_V1_FP.exists():
        sys.exit(
            f"ERROR: v1 pilot output missing at {PILOT_V1_FP}; cannot derive input_path_ids."
        )

    v1 = json.loads(PILOT_V1_FP.read_text(encoding="utf-8"))
    target_pids = set(v1.get("input_path_ids", []))
    if not target_pids:
        target_pids = {a["path_id"] for a in v1["raw_output"]["assignments"]}
    print(f"target path_ids from v1 pilot: {len(target_pids)}", flush=True)

    paths = load_deduped_paths()
    sample = [p for p in paths if p["path_id"] in target_pids]
    print(f"matched {len(sample)} / {len(target_pids)} pilot paths", flush=True)
    if len(sample) != len(target_pids):
        missing = target_pids - {p["path_id"] for p in sample}
        print(f"  WARNING: {len(missing)} missing: {sorted(missing)[:5]}", flush=True)

    print("loading node_attrs ...", flush=True)
    with open(M.STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    print(f"  {len(na)} nodes", flush=True)

    sentinel = uuid.uuid4().hex[:12]
    prompt = build_pilot_v2_prompt(sample, na, sentinel)
    print(f"prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)

    json_part, dur, _, err = M.streaming_call_with_validation(
        prompt,
        sentinel,
        "pilot_v2_100",
        PARTIAL_FP,
        model="claude-opus-4-7",
    )
    if err or not json_part:
        print(
            f"\nPILOT V2 FAILED ({err}). Partial preserved at {PARTIAL_FP.name}.",
            flush=True,
        )
        sys.exit(2)
    try:
        parsed = json.loads(json_part)
    except json.JSONDecodeError as e:
        print(f"\nJSON parse error: {e}", flush=True)
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
            "pilot_n": len(sample),
            "duration_sec": dur,
            "version": "v2_discovery_new_architecture",
            "v1_path_ids_source": PILOT_V1_FP.name,
            "n_axes": len(parsed.get("axes", [])),
            "n_harm_classes": len(parsed.get("harm_classes", [])),
            "n_mech_classes": len(parsed.get("mechanism_classes", [])),
            "n_assignments": len(parsed.get("assignments", [])),
            "raw_output": parsed,
            "input_path_ids": [p["path_id"] for p in sample],
        },
    )
    print(f"\nwrote {OUT_FP.name}", flush=True)
    print(f"  axes proposed:          {len(parsed.get('axes', []))}", flush=True)
    print(
        f"  harm_classes proposed:  {len(parsed.get('harm_classes', []))}", flush=True
    )
    print(
        f"  mech_classes proposed:  {len(parsed.get('mechanism_classes', []))}",
        flush=True,
    )
    print(f"  assignments:            {len(parsed.get('assignments', []))}", flush=True)

    # Quick eyeball on key signals
    asgs = parsed.get("assignments", [])
    n_hc_unassigned = sum(
        1
        for a in asgs
        if isinstance(a.get("harm_class_id"), dict)
        and a["harm_class_id"].get("unassigned")
    )
    n_mc_unassigned = sum(
        1
        for a in asgs
        if isinstance(a.get("mechanism_class_id"), dict)
        and a["mechanism_class_id"].get("unassigned")
    )
    fits = [a.get("fit_score") for a in asgs if a.get("fit_score") is not None]
    confs = [a.get("confidence") for a in asgs if a.get("confidence") is not None]
    print(
        f"\n  HC unassigned: {n_hc_unassigned}; MC unassigned: {n_mc_unassigned}",
        flush=True,
    )
    if fits:
        print(
            f"  fit_score: mean={sum(fits) / len(fits):.2f} "
            f"(low<=3: {sum(1 for f in fits if f <= 3)} / {len(fits)})",
            flush=True,
        )
    if confs:
        print(
            f"  confidence: mean={sum(confs) / len(confs):.2f} "
            f"(low<=3: {sum(1 for c in confs if c <= 3)} / {len(confs)})",
            flush=True,
        )

    if parsed.get("architecture_critique"):
        print(
            f"\narchitecture_critique:\n{parsed['architecture_critique']}", flush=True
        )


if __name__ == "__main__":
    main()
