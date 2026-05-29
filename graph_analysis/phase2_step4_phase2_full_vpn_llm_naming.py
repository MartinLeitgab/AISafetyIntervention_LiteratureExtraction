"""
phase2_step4_phase2_full_vpn_llm_naming.py — Phase 2 Task A FULL-VPN LLM CLUSTERING (§19.12)

LLM-central clustering of all 19,073 EDGE-only VPN nodes into mechanism classes.
HDBSCAN clustering becomes the validation check (§19.12.5), not the upstream substrate.

Key differences vs Pass-2:
  * Operates on ALL VPN nodes per pool (not just HDBSCAN-residuals)
  * System prompt explains the full 7-subtype reasoning chain (extraction-time roles)
    so the LLM understands WHERE in the chain each node sits and why subtype
    DIFFERENTIATES same-thought-different-function mechanisms
  * Subtype is METADATA on each node, NOT a clustering boundary
  * Granularity guidance: ~30-50 NR + ~15-25 risk classes; merge "same how, different where"
  * Atomic per-batch save (.tmp + os.replace) before next API call → no data loss on
    session-limit hit
  * AUP-resilience: on AUP block, auto-split batch into 8 sub-batches of 10 → save
    sub-batch outputs individually; failed sub-batches logged

Modes:
  --mode smoke   Run 5 batches, ONE per non-risk subtype (pa/ti/dr/im/va) for review.
                 Each batch is single-subtype to make per-subtype output behavior visible.
                 Cost: ~250k tokens, ~30min wall.
  --mode full    Run all 239 batches across NR pool + risk pool. Mixed-subtype batches.
                 Cost: ~11.8M tokens, ~10-15h wall. Requires explicit user approval.

Pools:
  --pool nr      Non-risk pool (16,609 nodes, 33 v3 seed groups)
  --pool risk    Risk pool (2,464 nodes, 24 v2 seed groups)
  --pool both    Run NR first, then risk

Outputs (per pool):
  phase2_full_vpn_batches/{pool}/batch_NNN.json   per-batch atomic saves
  phase2_full_vpn_decisions_{pool}.json           merged decisions
  cluster_memberships_rev8_paper_methodC_full_vpn_{pool}.pkl   class_name → [node_ids]
  phase2_full_vpn_summary_{pool}.json             counts + top classes
  phase2_full_vpn_aup_failures_{pool}.json        sub-batches that AUP-blocked even after split

Usage:
  python phase2_step4_phase2_full_vpn_llm_naming.py --mode smoke --pool nr
  python phase2_step4_phase2_full_vpn_llm_naming.py --mode full  --pool both
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

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 60-min subprocess timeout (per user instruction "no time cutoff")
os.environ.setdefault("CLAUDE_CLI_TIMEOUT_SEC", "3600")

SHIM_DIR = Path("C:/Users/malei/0_project_work/0_domain_finder/knowledge_pipeline/src")
sys.path.insert(0, str(SHIM_DIR))
from claude_cli_shim import ClaudeCLI  # noqa: E402

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
RUN_DIR_BASE = STEP1 / "phase2_full_vpn_batches"

# ---- Configuration ----
BATCH_SIZE = 80
SUB_BATCH_SIZE = 10  # AUP-fallback: split into sub-batches of this size
MAX_RETRIES_PER_BATCH = (
    1  # 1 retry on validation/AUP failure; on 2nd failure, sub-split
)
MAX_RETRIES_PER_SUB_BATCH = 2  # sub-batch can retry more aggressively
SMOKE_RNG_SEED = 20260509  # deterministic seed-gen sample selection (Pass A)
SMOKE_RNG_SEED_PHASE_B = (
    20260510  # distinct seed for Pass B smoke so it doesn't overlap with Pass A
)

# ---- Subtype canonical info (logical-chain order; per project CLAUDE.md) ----
# Verbatim role definitions from intervention_graph_creation/src/prompt/final_primary_prompt.py
SUBTYPE_CHAIN = [
    "risk",
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
    "intervention",
]
SUBTYPE_SHORT = {
    "risk": "risk",
    "problem_analysis": "pa",
    "theoretical_insight": "ti",
    "design_rationale": "dr",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
    "intervention": "interv",
}
SUBTYPE_DEFINITIONS = """\
Each input node was extracted from a single AI safety paper by an LLM following a strict
seven-step causal-interventional reasoning chain:

  risk → problem_analysis → theoretical_insight → design_rationale →
         implementation_mechanism → validation_evidence → intervention

The seven roles, in order:
  1. risk                     — "[Canonical Specific Phenomenon/Problem Name] in [Context]"
                                Top-level harms / undesired outcomes the paper addresses.
  2. problem_analysis (pa)    — "[Mechanism Causing Risk] in [Context]"
                                Decomposition of WHY the risk arises, characterized as
                                discrete causal mechanisms.
  3. theoretical_insight (ti) — "[Assumption / Hypothesized Resolution Opportunity] in [Context]"
                                Theoretical understanding, properties, theorems, or
                                hypothesized handles that suggest how the problem could
                                be addressed.
  4. design_rationale (dr)    — "[Solution Approach to Resolve Problem] in [Context]"
                                Design choices and high-level approach: WHY this strategy
                                is structured the way it is.
  5. implementation_mechanism — "[Technique/Implementation of Approach] in [Context]"
     (im)                       Concrete HOW: the specific technique by which the
                                approach is realized.
  6. validation_evidence (va) — "[Measurement and Result of Approach] in [Context]"
                                Empirical results, experiments, evaluations,
                                case-studies, ablations.
  7. intervention             — Action verb start; concrete intervention proposed by the
                                paper to reduce the risk.

CRITICAL — SUBTYPE IS A HARD CLUSTERING BOUNDARY:
- The SAME conceptual content serving DIFFERENT roles in the chain represents
  DIFFERENT mechanisms and MUST NOT be grouped together. Example: "human oversight
  of model outputs" appearing as a `design_rationale` (high-level safety strategy)
  vs as an `implementation_mechanism` (a specific deployment-time check) are TWO
  DIFFERENT mechanism families even though the surface text looks similar — they
  play different functional roles in the causal chain and warrant distinct clusters.
- Mechanism classes therefore live STRICTLY WITHIN A SINGLE SUBTYPE. Do not propose
  a class that spans multiple subtypes. Do not assign a node from subtype X into a
  class that was created from subtype-Y nodes.
- WITHIN a single subtype, group nodes that share the same underlying mechanism. Be
  willing to merge near-duplicate classes when the mechanistic difference is a
  matter of nuance (e.g., dataset choice or model size variant). But preserve genuine
  mechanistic distinctions when the "how" actually differs (e.g., representation
  engineering vs probing classifiers — both interpretability, but different levers).
- Default rule: when uncertain whether two within-subtype classes should merge, ask
  "do they target the SAME causal lever?" If yes → merge. If no (different lever
  even if same target) → keep separate. Do not over-split into singletons; do not
  over-coarsen across distinct levers.
"""

GRANULARITY_GUIDANCE = """\
GRANULARITY GUIDANCE (the most important constraint):

Your goal is to identify mechanism families that group single mechanism instantiations
by their fundamental commonalities and differences. The target catalog has roughly:
  - ~30-50 NR (non-risk) mechanism classes
  - ~15-25 risk mechanism classes

AVOID OVER-GRANULAR SPLITS. If two proposed classes share the same "how" (the
underlying mechanism by which an effect is achieved) and differ only in:
  - "where" (the subtype role they typically appear in), or
  - "specifics of one example" (which dataset, which model size, which task)
THEN MERGE them into one class. For example: "RL training mechanism: priority sampling"
+ "RL training mechanism: action-space augmentation" + "RL exploration mechanism:
human priors" should likely be ONE "RL training-loop modification mechanism" class,
not three.

DO NOT OVER-COARSEN either. Two classes that share surface vocabulary but differ in
the actual causal mechanism should remain separate. For example: "interpretability
mechanism: feature attribution" and "interpretability mechanism: representation
engineering" target different causal levers and should stay distinct.

DEFAULT TO THE SEED CATALOG. Propose new classes ONLY when the new theme is clearly
absent from the seed AND would absorb >= 5 nodes. Drift the seed names verbatim.
"""

NR_VERSION = "v3"
RISK_VERSION = "v2"

# Pass B consolidation cadence: every K=30 batches per subtype, scan residuals for
# clusters of >=5 similar nodes and propose new groups. Residual nodes from prior
# batches are RECONSIDERED — not treated as "done."
CONSOLIDATION_EVERY_K_BATCHES = 30
CONSOLIDATION_MIN_GROUP_SIZE = 5

# Pass A (seed-gen) per-subtype sample size and target group count.
# Target = ~50 families from 200-node sample (avg ~4 nodes/family). Pass B will extend
# the catalog with new families as more nodes are seen across the full ~2,500-node pool.
# The 0.1 * N_papers heuristic (~180/subtype) applies to the final TOTAL catalog after
# Pass B, NOT to Pass A alone.
SEED_SAMPLE_PER_SUBTYPE = 200
SEED_TARGET_GROUPS_PER_SUBTYPE = 50
SEED_MIN_GROUPS = 30
SEED_MAX_GROUPS = 80
SEED_MAX_TOKENS = 32768


def parse_emb(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        a = v.astype(np.float32)
    elif isinstance(v, str):
        s = v.strip().lstrip("<").rstrip(">")
        if not s:
            return None
        a = np.array([float(x) for x in s.split(",")], dtype=np.float32)
    else:
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


def truncate(s, n):
    return s if len(s) <= n else s[: n - 1] + "..."


def atomic_write_json(path: Path, obj):
    """Write JSON to path atomically: write .tmp, fsync, replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(obj, indent=2, default=str)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_seed_catalog(pool: str):
    """Load v3 NR or v2 risk seed groups."""
    if pool == "nr":
        path = STEP1 / "phase2_seed_taxonomy_nr_v3_recovered.json"
        with open(path, encoding="utf-8") as f:
            seed = json.load(f)
        return seed["parsed"]["groups"], NR_VERSION
    elif pool == "risk":
        path = STEP1 / "phase2_seed_taxonomy_risk_v2.json"
        with open(path, encoding="utf-8") as f:
            seed = json.load(f)
        return seed["parsed"]["groups"], RISK_VERSION
    else:
        raise ValueError(f"unknown pool {pool!r}")


def build_node_records(pool: str, role_of: dict, node_attrs: dict, restrict_ids=None):
    """Return list of all VPN node records for the pool, sorted by node_id.
    If restrict_ids is provided, only nodes whose id is IN restrict_ids are returned.
    """
    if pool == "nr":
        wanted = {
            st for st in SUBTYPE_CHAIN if st not in ("risk",)
        }  # 5 body + intervention
    elif pool == "risk":
        wanted = {"risk"}
    else:
        raise ValueError(f"unknown pool {pool!r}")
    records = []
    for nid, role in role_of.items():
        if role not in wanted:
            continue
        if restrict_ids is not None and int(nid) not in restrict_ids:
            continue
        a = node_attrs.get(nid) or node_attrs.get(int(nid)) or {}
        records.append(
            {
                "id": int(nid),
                "name": (a.get("name") or "").strip(),
                "description": (a.get("description") or "").strip(),
                "subtype": role,
            }
        )
    records.sort(key=lambda r: r["id"])
    return records


def load_active_catalog(pool: str):
    """Return {subtype: [list of {name, description, origin}]} — the live per-subtype
    catalog = Pass-A seeds + Pass-B 'new' groups + consolidation new-groups, deduped on name.
    """
    catalog = defaultdict(list)
    seen = defaultdict(set)
    subtypes = (
        [
            "problem_analysis",
            "theoretical_insight",
            "design_rationale",
            "implementation_mechanism",
            "validation_evidence",
            "intervention",
        ]
        if pool == "nr"
        else ["risk"]
    )
    # Pass-A seeds
    seed_dir = STEP1 / f"phase2_full_vpn_seed_per_subtype_{pool}"
    for st in subtypes:
        st_short = SUBTYPE_SHORT.get(st, st)
        seed_file = seed_dir / f"seed_{st_short}.json"
        if seed_file.exists():
            s = json.loads(seed_file.read_text(encoding="utf-8"))
            for g in s.get("groups", []):
                name = g.get("name", "")
                if name and name not in seen[st]:
                    seen[st].add(name)
                    catalog[st].append(
                        {
                            "name": name,
                            "description": g.get("description", ""),
                            "origin": "seed",
                        }
                    )
    # Pass-B 'new' groups proposed in prior batches
    run_dir = RUN_DIR_BASE / pool
    if run_dir.exists():
        for batch_file in sorted(run_dir.glob("batch_*.json")):
            try:
                saved = json.loads(batch_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for d in saved.get("decisions", []):
                if d.get("decision") == "new":
                    name = d.get("group_name", "")
                    st = d.get("subtype")
                    if name and st and name not in seen[st]:
                        seen[st].add(name)
                        catalog[st].append(
                            {
                                "name": name,
                                "description": d.get("group_description", ""),
                                "origin": "pass_b_new",
                            }
                        )
    # Consolidation new-groups
    cons_dir = STEP1 / f"phase2_full_vpn_consolidation_{pool}"
    if cons_dir.exists():
        for cons_file in sorted(cons_dir.glob("consolidation_*.json")):
            try:
                c = json.loads(cons_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for g in c.get("new_groups", []):
                name = g.get("name", "")
                st = g.get("subtype")
                if name and st and name not in seen[st]:
                    seen[st].add(name)
                    catalog[st].append(
                        {
                            "name": name,
                            "description": g.get("description", ""),
                            "origin": "consolidation",
                        }
                    )
    return dict(catalog)


def load_already_decided_ids(pool: str):
    """Nodes with decision in {seed, new} from any batch, PLUS consolidation reassignments.
    These are NOT reprocessed in regular batches.
    """
    decided = set()
    run_dir = RUN_DIR_BASE / pool
    if run_dir.exists():
        for batch_file in sorted(run_dir.glob("batch_*.json")):
            try:
                saved = json.loads(batch_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for d in saved.get("decisions", []):
                if (
                    d.get("decision") in ("seed", "new")
                    and d.get("node_id") is not None
                ):
                    decided.add(int(d["node_id"]))
    cons_dir = STEP1 / f"phase2_full_vpn_consolidation_{pool}"
    if cons_dir.exists():
        for cons_file in sorted(cons_dir.glob("consolidation_*.json")):
            try:
                c = json.loads(cons_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for r in c.get("reassignments", []):
                if r.get("node_id") is not None:
                    decided.add(int(r["node_id"]))
    return decided


def load_pending_residual_ids(pool: str):
    """Nodes marked residual in some batch and NOT yet absorbed by consolidation.
    These will be sent to consolidation pass — NOT re-batched."""
    residual = set()
    run_dir = RUN_DIR_BASE / pool
    if run_dir.exists():
        for batch_file in sorted(run_dir.glob("batch_*.json")):
            try:
                saved = json.loads(batch_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for d in saved.get("decisions", []):
                if d.get("decision") == "residual" and d.get("node_id") is not None:
                    residual.add(int(d["node_id"]))
    # Subtract absorbed
    decided = load_already_decided_ids(pool)
    return residual - decided


def load_hdbscan_clustered_node_ids(pool: str):
    """Return set of node IDs that are inside an HDBSCAN cluster for this pool.
    Pool 'nr' = NR-pool clusters (5 body subtypes + intervention).
    Pool 'risk' = risk-pool clusters.
    """
    with open(
        STEP1 / "cluster_memberships_rev8_paper_methodA_c75m3_subtype.pkl", "rb"
    ) as f:
        cm = pickle.load(f)
    target_pools = (
        {
            "problem_analysis",
            "theoretical_insight",
            "design_rationale",
            "implementation_mechanism",
            "validation_evidence",
            "intervention",
        }
        if pool == "nr"
        else {"risk"}
    )
    ids = set()
    for key, members in cm.items():
        if key[2] in target_pools:
            for m in members:
                ids.add(int(m))
    return ids


def make_seed_prompt(
    pool: str, subtype: str, records, target_groups: int, sentinel: str
):
    """Pass-A seed-generation prompt: propose ~target_groups mechanism families
    for THIS subtype based on a stratified-random sample of records.
    """
    body = "\n".join(fmt_node(i, r) for i, r in enumerate(records))
    n = len(records)
    return f"""You are PROPOSING the STARTING mechanism-family taxonomy for one role of an AI safety knowledge graph.

This is PASS A (seed generation), the FIRST of TWO LLM passes:
  - Pass A (this call): see {n} stratified-random nodes from the {subtype} pool. Propose a STARTING catalog of mechanism families.
  - Pass B (later): see all ~2,500-3,000 nodes from this subtype pool. Extend the catalog with NEW families when needed; consolidation passes can also merge near-duplicate families.

Your job here is to propose a STARTING catalog that is:
  - Tight enough to give Pass B clear assignment targets (avg ~4 nodes/family in this 200-node sample = ~{target_groups} families).
  - Loose enough that Pass B has room to add roughly 2-4× more families when the rest of the corpus reveals additional mechanisms.
  - NOT a complete enumeration of all mechanisms in the corpus (you only see ~7% of the pool here).

================================================================
SUBTYPE CONTEXT — the seven roles in the extraction reasoning chain
================================================================

{SUBTYPE_DEFINITIONS}

================================================================
TARGET FOR THIS CALL: subtype = {subtype}
================================================================

Empirical guidance for THIS call:
  - Aim for ~{target_groups} mechanism families.
  - Acceptable range: {SEED_MIN_GROUPS}-{SEED_MAX_GROUPS}. Below {SEED_MIN_GROUPS} = over-coarse; above {SEED_MAX_GROUPS} = over-granular for a 200-node sample.
  - Each family in this catalog will see roughly 50-100 additional members in Pass B (final family sizes ~50-100 nodes after assignment of full ~2,500-node pool).

Propose mechanism-family names that:
- Are mechanism-centric (the "how" by which an effect is achieved), NOT topic-centric (the "what is studied").
- Use concrete naming: "RL training: priority sampling from replay buffers" (good); "RL stuff" (bad).
- Span across papers but stay tight enough that two researchers reading the name+description would agree which nodes belong.
- Include a one-line description (~25-40 words) for each family.

OUTPUT FORMAT — STRICT (validation will reject malformed responses):
- Output ONLY one JSON object. No preamble, no markdown fences.
- Start with `{{`. End with the closing `}}` then `END_SENTINEL_{sentinel}` on the same line.

Schema:

{{
  "subtype": "{subtype}",
  "groups": [
    {{"name": "<concrete mechanism family name>", "description": "<one-line ~30-word description>"}},
    ...
  ]
}}END_SENTINEL_{sentinel}

INPUT NODES ({n} stratified-random samples from the {subtype} pool; format: `(subtype) name -- description`):

{body}

Now produce the STARTING seed catalog ({SEED_MIN_GROUPS}-{SEED_MAX_GROUPS} groups, target ~{target_groups})."""


def parse_seed_response(text):
    if text is None:
        return None
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except Exception as e:
        # Try to recover groups via regex
        print(f"    seed parse failed: {e}; trying regex recovery")
        patt = re.compile(
            r'\{\s*"name":\s*"((?:[^"\\]|\\.)*)"\s*,\s*"description":\s*"((?:[^"\\]|\\.)*)"\s*\}',
            re.DOTALL,
        )
        groups = [
            {"name": m.group(1), "description": m.group(2)} for m in patt.finditer(t)
        ]
        if groups:
            print(f"    recovered {len(groups)} groups via regex")
            return {"groups": groups, "_recovered_via_regex": True}
        return None


def run_seed_gen(pool: str, role_of: dict, node_attrs: dict):
    """Pass A: per-subtype seed generation.
    Restricted to HDBSCAN-clustered nodes only (1:1 with HDBSCAN scope).
    """
    print("=" * 80)
    print(f"PASS A — SEED GENERATION (pool={pool})")
    print("=" * 80)

    in_cluster_ids = load_hdbscan_clustered_node_ids(pool)
    print(f"HDBSCAN-clustered node count for pool={pool}: {len(in_cluster_ids)}")

    all_records = build_node_records(
        pool, role_of, node_attrs, restrict_ids=in_cluster_ids
    )
    print(f"records (in HDBSCAN clusters, pool={pool}): {len(all_records)}")

    out_dir = STEP1 / f"phase2_full_vpn_seed_per_subtype_{pool}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine subtypes
    if pool == "nr":
        subtypes = [
            "problem_analysis",
            "theoretical_insight",
            "design_rationale",
            "implementation_mechanism",
            "validation_evidence",
            "intervention",
        ]
    else:
        subtypes = ["risk"]

    rng = random.Random(SMOKE_RNG_SEED)
    per_subtype_seeds = {}

    for st in subtypes:
        out_file = out_dir / f"seed_{SUBTYPE_SHORT.get(st, st)}.json"
        if out_file.exists():
            print(f"\n[idempotent skip] {out_file.name} already done")
            saved = json.loads(out_file.read_text(encoding="utf-8"))
            per_subtype_seeds[st] = saved
            continue

        print(f"\n=== seed-gen for subtype={st} ===")
        st_records = [r for r in all_records if r["subtype"] == st]
        print(f"  {st} pool size (in clusters): {len(st_records)}")
        if len(st_records) == 0:
            continue
        sample_size = min(SEED_SAMPLE_PER_SUBTYPE, len(st_records))
        sample = rng.sample(st_records, sample_size)
        sentinel = uuid.uuid4().hex[:12]
        prompt = make_seed_prompt(
            pool, st, sample, SEED_TARGET_GROUPS_PER_SUBTYPE, sentinel
        )
        print(
            f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens), max_tokens={SEED_MAX_TOKENS}"
        )

        # Call with custom max_tokens
        client = ClaudeCLI()
        end_marker = f"END_SENTINEL_{sentinel}"
        json_part = None
        for attempt in range(MAX_RETRIES_PER_BATCH + 1):
            print(f"  attempt {attempt + 1}/{MAX_RETRIES_PER_BATCH + 1} ...")
            t0 = time.time()
            try:
                resp = client.messages.create(
                    model="claude-opus-4-7",
                    system=(
                        "You produce STRICT JSON taxonomy output. Never preamble, "
                        "never use markdown fences, always emit valid JSON, always end "
                        "with the requested sentinel."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=SEED_MAX_TOKENS,
                )
                text = resp.content[0].text
                duration = time.time() - t0
                print(f"    returned {len(text)} chars in {duration:.0f}s")
                trimmed = text.strip()
                if trimmed.startswith("{") and trimmed.endswith(end_marker):
                    print("    OK validation")
                    json_part = trimmed[: -len(end_marker)].rstrip()
                    break
                else:
                    print(
                        f"    FAIL validation; first/last 100: {repr(trimmed[:100])} ... {repr(trimmed[-100:])}"
                    )
            except Exception as e:
                print(f"    FAIL shim error: {type(e).__name__}: {str(e)[:200]}")

        parsed = parse_seed_response(json_part)
        if parsed and "groups" in parsed:
            n_groups = len(parsed["groups"])
            print(f"  -> {st}: {n_groups} groups proposed")
            saved_obj = {
                "pool": pool,
                "subtype": st,
                "n_input_nodes": len(sample),
                "target_groups": SEED_TARGET_GROUPS_PER_SUBTYPE,
                "n_groups_proposed": n_groups,
                "input_node_ids": [r["id"] for r in sample],
                "groups": parsed["groups"],
                "recovered_via_regex": parsed.get("_recovered_via_regex", False),
            }
            atomic_write_json(out_file, saved_obj)
            per_subtype_seeds[st] = saved_obj
        else:
            print(f"  -> {st}: FAILED")

    # Final summary
    merged_path = STEP1 / f"phase2_full_vpn_seed_per_subtype_{pool}_summary.json"
    summary = {
        "pool": pool,
        "n_subtypes": len(per_subtype_seeds),
        "total_groups_across_subtypes": sum(
            s["n_groups_proposed"] for s in per_subtype_seeds.values()
        ),
        "per_subtype_counts": {
            st: s["n_groups_proposed"] for st, s in per_subtype_seeds.items()
        },
    }
    atomic_write_json(merged_path, summary)
    print("\n" + "=" * 80)
    print("SEED GENERATION COMPLETE")
    print("=" * 80)
    for st, n in summary["per_subtype_counts"].items():
        print(f"  {st:<28}: {n} groups")
    print(f"  TOTAL across subtypes: {summary['total_groups_across_subtypes']}")
    print(
        "\nNext step: run merge/dedup script to consolidate cross-subtype seed catalog."
    )


def stratified_smoke_sample(records, batch_size, rng):
    """For smoke test: 1 batch per NR subtype, single-subtype, batch_size each.
    Returns dict subtype -> list[record].
    """
    by_st = defaultdict(list)
    for r in records:
        if r["subtype"] != "risk":
            by_st[r["subtype"]].append(r)
    out = {}
    for st in [
        "problem_analysis",
        "theoretical_insight",
        "design_rationale",
        "implementation_mechanism",
        "validation_evidence",
        "intervention",
    ]:
        pool = by_st.get(st, [])
        if len(pool) <= batch_size:
            out[st] = list(pool)
        else:
            out[st] = rng.sample(pool, batch_size)
    return out


def build_seed_catalog_block(seed_groups):
    lines = []
    for i, g in enumerate(seed_groups):
        name = g.get("name") or g.get("group_name") or f"group_{i}"
        desc = g.get("description") or g.get("desc") or ""
        lines.append(f"  G{i + 1}. {name}\n      desc: {truncate(desc, 200)}")
    return "\n".join(lines)


def fmt_node(i, r):
    st = SUBTYPE_SHORT.get(r["subtype"], r["subtype"])
    return (
        f"{i}. ({st}) {truncate(r['name'], 100)} -- {truncate(r['description'], 250)}"
    )


def make_prompt(pool, records, seed_groups, sentinel):
    seed_catalog = build_seed_catalog_block(seed_groups)
    body = "\n".join(fmt_node(i, r) for i, r in enumerate(records))
    n = len(records)
    pool_label = "non-risk (5 body subtypes + intervention)" if pool == "nr" else "risk"

    return f"""You are clustering AI safety literature concept/intervention nodes into mechanism classes.

This is the FULL-VPN LLM clustering pass for the {pool_label} pool. Every node was extracted from a single AI safety paper by an LLM following a strict reasoning chain (described below). Your task: assign each input node to ONE mechanism class.

================================================================
SUBTYPE CONTEXT — the seven roles in the extraction reasoning chain
================================================================

{SUBTYPE_DEFINITIONS}

================================================================
{GRANULARITY_GUIDANCE}
================================================================

OUTPUT FORMAT — STRICT (validation will reject malformed responses):
- Output ONLY one JSON object. No preamble, no markdown fences, no commentary.
- Start your output with the character `{{`.
- After the closing `}}`, append the literal sentinel `END_SENTINEL_{sentinel}` on the same line.

DECISION OPTIONS PER NODE (pick exactly one):
- `{{"index": N, "decision": "seed", "group_name": "<verbatim group name from catalog below>", "confidence": "high"|"medium"|"low"}}` — fold into one of the {len(seed_groups)} seed catalog classes. group_name MUST match a seed name VERBATIM — do NOT prepend the display index (e.g., NOT "G7. ..." — just the name itself, exactly as shown after `Gnn.` in the catalog).
- `{{"index": N, "decision": "new", "group_name": "<your proposed new class name>", "group_description": "<1-2 sentence description>", "confidence": "high"|"medium"}}` — propose a NEW class. Use ONLY when the node's mechanism is clearly absent from the seed catalog AND you expect >= 5 nodes will fit this new class. Re-use the SAME new group_name verbatim across multiple nodes in the same batch if they belong together.
- `{{"index": N, "decision": "residual"}}` — node does not fit any existing class AND you cannot yet justify a NEW class (e.g., only 1-2 nodes in batch with this mechanism). Residuals are NOT wasted — they accumulate across batches and a periodic CONSOLIDATION pass groups clusters of >= 5 similar residuals into new mechanism families. Use residual freely when uncertain; do not force a fit.

GUIDANCE:
- Subtype is a HARD CLUSTERING BOUNDARY (see SUBTYPE CONTEXT above). All batches are single-subtype, so all catalog groups shown here belong to the batch's subtype. NEVER propose a `new` class that mixes subtypes.
- Within the matching subtype, mechanism family (the "how") takes priority over surface keyword similarity.
- Default to the seed catalog. Propose `new` only when truly required (no existing within-subtype class fits AND >= 5 similar nodes likely in pool).
- When 2 within-subtype classes look near-duplicate but target the SAME causal lever → fold to whichever one has clearer naming. When they target DIFFERENT levers (even if same target) → keep separate.

SEED CATALOG ({len(seed_groups)} mechanism classes):

{seed_catalog}

INPUT NODES ({n} nodes indexed 0 to {n - 1}; format: `(subtype) name -- description`):

{body}

OUTPUT — single JSON object, no preamble, end with `END_SENTINEL_{sentinel}`:

{{
  "node_decisions": [
    {{"index": 0, "decision": "...", ...}},
    {{"index": 1, "decision": "...", ...}}
  ]
}}END_SENTINEL_{sentinel}

Now produce the response."""


def call_with_validation(prompt, sentinel, label, max_retries=MAX_RETRIES_PER_BATCH):
    """Call shim; validate sentinel + start-marker; retry on failure.
    Returns (json_part_or_None, duration_sec, attempts_used, error_kind).
    error_kind ∈ {None, "aup", "validation", "shim", "exception"}.
    """
    client = ClaudeCLI()
    end_marker = f"END_SENTINEL_{sentinel}"
    duration = 0.0
    last_error_kind = "shim"
    for attempt in range(max_retries + 1):
        print(f"  [{label}] attempt {attempt + 1}/{max_retries + 1} ...")
        t0 = time.time()
        try:
            resp = client.messages.create(
                model="claude-opus-4-7",
                system=(
                    "You produce STRICT JSON output for a mechanism-family clustering "
                    "pipeline. Never preamble, never use markdown fences, always emit "
                    "valid JSON, always end your output with the requested sentinel."
                ),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
            )
            text = resp.content[0].text
            duration = time.time() - t0
            print(f"    returned {len(text)} chars in {duration:.0f}s")
            trimmed = text.strip()
            ok_start = trimmed.startswith("{")
            ok_end = trimmed.endswith(end_marker)
            if ok_start and ok_end:
                print("    OK start-marker + sentinel both present")
                json_part = trimmed[: -len(end_marker)].rstrip()
                return json_part, duration, attempt + 1, None
            else:
                last_error_kind = "validation"
                print(f"    FAIL validation (start={ok_start}, end={ok_end})")
                print(f"      first 100 chars: {repr(trimmed[:100])}")
                print(f"      last 100 chars: {repr(trimmed[-100:])}")
        except Exception as e:
            duration = time.time() - t0
            err_str = str(e)
            print(f"    FAIL shim error: {type(e).__name__}: {err_str[:200]}")
            if (
                "Usage Policy" in err_str
                or "AUP" in err_str
                or "violate our Usage" in err_str
            ):
                last_error_kind = "aup"
                # AUP is reproducible; do not retry the same batch — caller will sub-split
                return None, duration, attempt + 1, "aup"
            last_error_kind = "shim"
    return None, duration, max_retries + 1, last_error_kind


def parse_with_fallback(text, batch_size):
    if text is None:
        return None
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except Exception as e:
        print(f"    parse failed: {e}; trying regex fallback")
        patt = re.compile(
            r'\{"index":\s*(\d+),\s*"decision":\s*"(seed|new|residual)"'
            r'(?:,\s*"group_name":\s*"((?:[^"\\]|\\.)*)")?'
            r'(?:,\s*"group_description":\s*"((?:[^"\\]|\\.)*)")?'
            r'(?:,\s*"confidence":\s*"([^"]*)")?\}',
            re.DOTALL,
        )
        decisions = []
        for m in patt.finditer(t):
            d = {"index": int(m.group(1)), "decision": m.group(2)}
            if m.group(3):
                d["group_name"] = m.group(3)
            if m.group(4):
                d["group_description"] = m.group(4)
            if m.group(5):
                d["confidence"] = m.group(5)
            decisions.append(d)
        if decisions:
            print(f"    regex fallback recovered {len(decisions)} of {batch_size}")
            return {"node_decisions": decisions, "_recovered_via_regex": True}
        return None


def run_one_batch(records, seed_groups, pool, label, max_retries):
    """Run a single batch (or sub-batch) end-to-end. Returns parsed decisions or None."""
    sentinel = uuid.uuid4().hex[:12]
    prompt = make_prompt(pool, records, seed_groups, sentinel)
    print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)")
    json_part, duration, attempts, err_kind = call_with_validation(
        prompt, sentinel, label, max_retries=max_retries
    )
    parsed = parse_with_fallback(json_part, len(records)) if json_part else None
    return parsed, duration, attempts, err_kind


_G_PREFIX_RE = re.compile(r"^G\d+\.\s+")


def _strip_g_prefix(name):
    if not name:
        return name
    return _G_PREFIX_RE.sub("", name.strip())


def translate_decisions(decisions, batch_records):
    """Local index -> global node info. Strips G-prefix that LLM sometimes prepends to group_name."""
    out = []
    for d in decisions:
        local_i = d.get("index")
        if local_i is None or not (0 <= local_i < len(batch_records)):
            continue
        rec = batch_records[local_i]
        out.append(
            {
                "node_id": rec["id"],
                "name": rec.get("name", ""),
                "subtype": rec["subtype"],
                "decision": d.get("decision"),
                "group_name": _strip_g_prefix(d.get("group_name")),
                "group_description": d.get("group_description"),
                "confidence": d.get("confidence"),
            }
        )
    return out


def process_batch_with_aup_resilience(
    batch, seed_groups, pool, batch_idx, run_dir, aup_failures
):
    """Run a batch; on AUP block, sub-split into chunks of SUB_BATCH_SIZE.
    Returns list of decision dicts (translated) — possibly empty.
    """
    label = f"{pool}_batch_{batch_idx:03d}"
    parsed, duration, attempts, err_kind = run_one_batch(
        batch, seed_groups, pool, label, max_retries=MAX_RETRIES_PER_BATCH
    )
    if parsed and "node_decisions" in parsed:
        return (
            translate_decisions(parsed["node_decisions"], batch),
            {
                "status": "ok",
                "duration_sec": round(duration, 1),
                "attempts": attempts,
                "recovered_via_regex": parsed.get("_recovered_via_regex", False),
            },
            "ok",
        )
    if err_kind == "aup":
        # Sub-split
        print(
            f"  AUP block on full batch; sub-splitting into chunks of {SUB_BATCH_SIZE}"
        )
        sub_decisions = []
        sub_meta = []
        n_sub = (len(batch) + SUB_BATCH_SIZE - 1) // SUB_BATCH_SIZE
        for sub_i in range(n_sub):
            sub_start = sub_i * SUB_BATCH_SIZE
            sub_end = min(sub_start + SUB_BATCH_SIZE, len(batch))
            sub_batch = batch[sub_start:sub_end]
            sub_label = f"{label}_sub{sub_i:02d}"
            sub_parsed, sub_dur, sub_att, sub_err = run_one_batch(
                sub_batch,
                seed_groups,
                pool,
                sub_label,
                max_retries=MAX_RETRIES_PER_SUB_BATCH,
            )
            if sub_parsed and "node_decisions" in sub_parsed:
                sub_decisions.extend(
                    translate_decisions(sub_parsed["node_decisions"], sub_batch)
                )
                sub_meta.append(
                    {
                        "sub_idx": sub_i,
                        "status": "ok",
                        "duration_sec": round(sub_dur, 1),
                        "attempts": sub_att,
                        "n_input": len(sub_batch),
                    }
                )
            else:
                # Sub-batch still failed; log node IDs for manual review
                aup_failures.append(
                    {
                        "batch_idx": batch_idx,
                        "sub_idx": sub_i,
                        "node_ids": [r["id"] for r in sub_batch],
                        "node_subtypes": [r["subtype"] for r in sub_batch],
                        "node_names": [truncate(r["name"], 80) for r in sub_batch],
                        "err_kind": sub_err,
                    }
                )
                sub_meta.append(
                    {
                        "sub_idx": sub_i,
                        "status": "failed",
                        "duration_sec": round(sub_dur, 1),
                        "attempts": sub_att,
                        "n_input": len(sub_batch),
                        "err_kind": sub_err,
                    }
                )
        return (
            sub_decisions,
            {
                "status": "ok_via_subsplit" if sub_decisions else "failed",
                "n_sub_batches": n_sub,
                "sub_meta": sub_meta,
            },
            "subsplit",
        )
    # Non-AUP failure: leave for retry on next run (do not save batch file)
    return (
        [],
        {
            "status": "failed",
            "duration_sec": round(duration, 1),
            "attempts": attempts,
            "err_kind": err_kind,
        },
        "failed",
    )


def run_pool(
    pool: str,
    mode: str,
    role_of: dict,
    node_attrs: dict,
    scope: str = "hdbscan_clustered",
):
    """Pass B (mode=full) — single-subtype batches, per-subtype catalog from Pass A,
    residual carry-forward via periodic consolidation. New-group proposal allowed.

    scope:
      "hdbscan_clustered" — restrict to HDBSCAN-clustered nodes (Pass B run 1; 1:1 with §19.12.5)
      "all_vpn"           — full VPN pool (Pass B run 2; processes residual delta on top of run 1)
    """
    print("=" * 80)
    print(f"FULL-VPN LLM clustering — pool={pool}, mode={mode}, scope={scope}")
    print("=" * 80)

    in_cluster_ids = load_hdbscan_clustered_node_ids(pool)
    print(f"HDBSCAN-clustered node count for pool={pool}: {len(in_cluster_ids)}")

    if scope == "hdbscan_clustered":
        all_records = build_node_records(
            pool, role_of, node_attrs, restrict_ids=in_cluster_ids
        )
        print(f"scope=hdbscan_clustered: pool node count = {len(all_records)}")
    elif scope == "all_vpn":
        all_records = build_node_records(pool, role_of, node_attrs, restrict_ids=None)
        n_residual = sum(1 for r in all_records if r["id"] not in in_cluster_ids)
        print(
            f"scope=all_vpn: pool node count = {len(all_records)} "
            f"({len(all_records) - n_residual} clustered + {n_residual} HDBSCAN-residual)"
        )
    else:
        raise ValueError(f"unknown scope {scope!r}")

    # Single output dir per pool
    run_dir = RUN_DIR_BASE / pool
    run_dir.mkdir(parents=True, exist_ok=True)
    aup_path = STEP1 / f"phase2_full_vpn_aup_failures_{pool}.json"
    aup_failures = []
    if aup_path.exists():
        try:
            aup_failures = json.loads(aup_path.read_text(encoding="utf-8"))
        except Exception:
            aup_failures = []

    # Build batches: SINGLE-SUBTYPE batching aligned with subtype-as-hard-boundary.
    # Residual decisions are NOT treated as "done" — residuals get reprocessed via
    # the periodic consolidation pass (every K=CONSOLIDATION_EVERY_K_BATCHES).
    already_decided_ids = load_already_decided_ids(pool)
    pending_residuals = load_pending_residual_ids(pool)
    print(
        f"resume: {len(already_decided_ids)} nodes already seed|new-decided (skip in batches)"
    )
    print(
        f"        {len(pending_residuals)} nodes pending residual (will be reprocessed via consolidation)"
    )

    if mode == "smoke":
        # Smoke kept for back-compat; uses 1 batch per NR subtype from prior design.
        if pool == "risk":
            print("(smoke mode is NR-only by design; skipping risk pool)")
            return
        rng = random.Random(SMOKE_RNG_SEED_PHASE_B)
        # Smoke = 1 batch per NR subtype, never re-pick already-decided
        unfinished = [
            r
            for r in all_records
            if r["id"] not in already_decided_ids and r["id"] not in pending_residuals
        ]
        smoke_sample = stratified_smoke_sample(unfinished, BATCH_SIZE, rng)
        batches = []
        for st in [
            "problem_analysis",
            "theoretical_insight",
            "design_rationale",
            "implementation_mechanism",
            "validation_evidence",
            "intervention",
        ]:
            recs = smoke_sample.get(st, [])
            if recs:
                batches.append((f"smoke_{SUBTYPE_SHORT.get(st, st)}", recs, st))
        print(
            f"smoke: {len(batches)} single-subtype batches "
            f"({sum(len(b[1]) for b in batches)} nodes total)"
        )
    else:
        # Pass B (full mode): single-subtype batches.
        # Skip nodes that are already-decided OR currently-pending-residual
        # (residuals will be handled by consolidation, not re-batched).
        skip_set = already_decided_ids | pending_residuals
        subtypes = (
            [
                "problem_analysis",
                "theoretical_insight",
                "design_rationale",
                "implementation_mechanism",
                "validation_evidence",
                "intervention",
            ]
            if pool == "nr"
            else ["risk"]
        )
        batches = []
        for st in subtypes:
            st_records = [
                r for r in all_records if r["subtype"] == st and r["id"] not in skip_set
            ]
            st_short = SUBTYPE_SHORT.get(st, st)
            for i in range(0, len(st_records), BATCH_SIZE):
                chunk = st_records[i : i + BATCH_SIZE]
                batches.append((f"{st_short}_{i // BATCH_SIZE:03d}", chunk, st))
        print(
            f"full: {len(batches)} single-subtype batches across {len(subtypes)} subtypes"
        )
        # Per-subtype breakdown
        from collections import Counter

        st_counts = Counter(b[2] for b in batches)
        for st, n in sorted(st_counts.items()):
            print(f"  {st:<28}: {n} batches")

    all_decisions = []
    n_failed = 0
    n_subsplit = 0
    n_batches_completed_in_run = 0  # for consolidation cadence

    # Load active catalog per subtype (will refresh after each consolidation)
    catalog_by_subtype = load_active_catalog(pool)
    print("active catalog by subtype:")
    for st, gs in catalog_by_subtype.items():
        print(f"  {st:<28}: {len(gs)} groups")

    for batch_idx, (batch_label, batch, batch_subtype) in enumerate(batches):
        # File name encodes label so smoke + full names don't collide
        batch_file = run_dir / f"batch_{batch_label}.json"
        # Idempotent skip
        if batch_file.exists():
            print(f"\n[idempotent skip] {batch_file.name} already done")
            saved = json.loads(batch_file.read_text(encoding="utf-8"))
            all_decisions.extend(saved.get("decisions", []))
            continue

        # Trigger consolidation every K batches WITHIN this run (and at end)
        if (
            n_batches_completed_in_run > 0
            and n_batches_completed_in_run % CONSOLIDATION_EVERY_K_BATCHES == 0
        ):
            iter_num = n_batches_completed_in_run // CONSOLIDATION_EVERY_K_BATCHES
            print(
                f"\n--- consolidation trigger (iter={iter_num}) after {n_batches_completed_in_run} batches ---"
            )
            run_consolidation(pool, role_of, node_attrs, iter_num)
            # Refresh active catalog after consolidation
            catalog_by_subtype = load_active_catalog(pool)
            for st, gs in catalog_by_subtype.items():
                print(f"  post-consolidation {st:<28}: {len(gs)} groups")

        print(
            f"\n=== batch {batch_idx:03d}/{len(batches) - 1}: {batch_label} "
            f"(subtype={batch_subtype}, {len(batch)} nodes) ==="
        )
        # Use per-subtype catalog for this batch (single-subtype batches)
        st_catalog = catalog_by_subtype.get(batch_subtype, [])
        decisions, meta, status = process_batch_with_aup_resilience(
            batch, st_catalog, pool, batch_idx, run_dir, aup_failures
        )

        # Save per-batch atomically BEFORE next batch starts
        batch_out = {
            "batch_idx": batch_idx,
            "batch_label": batch_label,
            "scope": scope,
            "n_input": len(batch),
            "n_decisions": len(decisions),
            "node_ids_global": [r["id"] for r in batch],
            "subtype_distribution": dict(
                {
                    st: sum(1 for r in batch if r["subtype"] == st)
                    for st in set(r["subtype"] for r in batch)
                }
            ),
            **meta,
            "decisions": decisions,
        }
        atomic_write_json(batch_file, batch_out)
        # Persist AUP failures continuously too
        if aup_failures:
            atomic_write_json(aup_path, aup_failures)

        if status == "failed":
            n_failed += 1
        elif status == "subsplit":
            n_subsplit += 1
        all_decisions.extend(decisions)
        n_batches_completed_in_run += 1

        n_seed = sum(1 for d in decisions if d["decision"] == "seed")
        n_new = sum(1 for d in decisions if d["decision"] == "new")
        n_res = sum(1 for d in decisions if d["decision"] == "residual")
        print(
            f"  -> batch {batch_idx:03d}: {len(decisions)} decisions "
            f"({n_seed} seed, {n_new} new, {n_res} residual) [{status}]"
        )

    # Final consolidation pass at end of run (always)
    if mode == "full":
        iter_num = (n_batches_completed_in_run // CONSOLIDATION_EVERY_K_BATCHES) + 1
        print(f"\n--- FINAL consolidation pass (iter={iter_num}) ---")
        run_consolidation(pool, role_of, node_attrs, iter_num)

    # ---- Final merge: load ALL batch files in run_dir (smoke + full combined) ----
    print("\n" + "=" * 80)
    print(f"POOL {pool} ({mode}) COMPLETE — merging ALL batches in {run_dir}")
    print("=" * 80)
    all_decisions = []
    for batch_file in sorted(run_dir.glob("batch_*.json")):
        try:
            saved = json.loads(batch_file.read_text(encoding="utf-8"))
            all_decisions.extend(saved.get("decisions", []))
        except Exception as e:
            print(f"  WARN: failed to load {batch_file.name}: {e}")
    print(f"failed batches: {n_failed}  /  subsplit batches: {n_subsplit}")
    print(f"total decisions: {len(all_decisions)}")
    print(f"AUP-blocked sub-batches needing manual review: {len(aup_failures)}")

    combined = {
        "pool": pool,
        "mode": mode,
        "version": "full_vpn_per_subtype_catalog",
        "n_input_total": len(all_records),
        "n_decisions": len(all_decisions),
        "n_failed_batches": n_failed,
        "n_subsplit_batches": n_subsplit,
        "n_aup_failed_subbatches": len(aup_failures),
        "summary": {
            "n_seed_assigned": sum(1 for d in all_decisions if d["decision"] == "seed"),
            "n_new_class_assigned": sum(
                1 for d in all_decisions if d["decision"] == "new"
            ),
            "n_residual": sum(1 for d in all_decisions if d["decision"] == "residual"),
        },
        "decisions": all_decisions,
    }
    decisions_path = STEP1 / f"phase2_full_vpn_decisions_{pool}_{mode}.json"
    atomic_write_json(decisions_path, combined)
    print(f"saved {decisions_path.name}")

    # cluster_memberships PKL
    method_c = defaultdict(list)
    for d in all_decisions:
        if d["decision"] in ("seed", "new") and d.get("group_name"):
            key = ("rev8_paper", "llm_full_vpn", pool, "methodC", d["group_name"])
            method_c[key].append(d["node_id"])
    pkl_path = (
        STEP1 / f"cluster_memberships_rev8_paper_methodC_full_vpn_{pool}_{mode}.pkl"
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(dict(method_c), f)
    print(
        f"saved {pkl_path.name}: {len(method_c)} classes / "
        f"{sum(len(v) for v in method_c.values())} members"
    )

    group_sizes = sorted(
        [(name[4], len(members)) for name, members in method_c.items()],
        key=lambda x: -x[1],
    )
    summary = {
        **combined["summary"],
        "n_classes_used": len(method_c),
        "n_new_classes_proposed": len(
            set(
                d["group_name"]
                for d in all_decisions
                if d["decision"] == "new" and d.get("group_name")
            )
        ),
        "top_classes_by_size": group_sizes[:20],
        "n_failed_batches": n_failed,
        "n_subsplit_batches": n_subsplit,
        "n_aup_failed_subbatches": len(aup_failures),
    }
    atomic_write_json(STEP1 / f"phase2_full_vpn_summary_{pool}_{mode}.json", summary)

    print("\nFINAL SUMMARY:")
    for k, v in summary.items():
        if k != "top_classes_by_size":
            print(f"  {k}: {v}")
    print("  top classes by member count:")
    for name, n in group_sizes[:15]:
        print(f"    [{n}] {truncate(name, 90)}")
    print("\n" + "=" * 80)
    print(f"POOL {pool} ({mode}) DONE")
    print("=" * 80)


def make_consolidation_prompt(subtype: str, residual_records, sentinel: str):
    """Consolidation: cluster N residual nodes into NEW mechanism groups of >=5 each."""
    body = "\n".join(fmt_node(i, r) for i, r in enumerate(residual_records))
    n = len(residual_records)
    return f"""You are CONSOLIDATING residual nodes from subtype={subtype} into NEW mechanism-family groups.

These {n} nodes were marked residual by prior Pass-B batches because no then-existing catalog group fit. Now you see them all together. Identify clusters of >= {CONSOLIDATION_MIN_GROUP_SIZE} similar residual nodes that share an underlying mechanism, propose NEW mechanism-family groups for those clusters, and assign each cluster's members.

================================================================
SUBTYPE CONTEXT
================================================================

{SUBTYPE_DEFINITIONS}

================================================================
TASK
================================================================

CONSTRAINTS:
- Subtype is HARD BOUNDARY: all proposed groups belong to subtype={subtype}. Do not create cross-subtype groups.
- Each new group MUST have >= {CONSOLIDATION_MIN_GROUP_SIZE} members. If fewer would fit, do NOT propose.
- Mechanism-centric naming (the "how" by which an effect is achieved). Concrete names.
- A node that does NOT cluster with >= {CONSOLIDATION_MIN_GROUP_SIZE - 1} others stays residual.
- Be willing to leave many nodes residual — only consolidate genuine clusters. Singletons stay residual.

OUTPUT FORMAT — STRICT:
- One JSON object. No preamble, no markdown.
- Start with `{{`. End with closing `}}` then `END_SENTINEL_{sentinel}`.

Schema:

{{
  "subtype": "{subtype}",
  "new_groups": [
    {{"name": "<concrete mechanism family>", "description": "<~30-word description>"}}
  ],
  "reassignments": [
    {{"node_index": <int>, "new_group_name": "<verbatim name from new_groups above>"}}
  ]
}}END_SENTINEL_{sentinel}

INPUT RESIDUAL NODES ({n} nodes; format: `(subtype) name -- description`):

{body}

Now produce the consolidation."""


def parse_consolidation_response(text):
    if text is None:
        return None
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except Exception as e:
        print(f"    consolidation parse failed: {e}")
        return None


def run_consolidation(pool: str, role_of: dict, node_attrs: dict, iteration: int):
    """Per-subtype consolidation pass: collect residuals, propose new groups for clusters of >=5."""
    cons_dir = STEP1 / f"phase2_full_vpn_consolidation_{pool}"
    cons_dir.mkdir(parents=True, exist_ok=True)
    out_file = cons_dir / f"consolidation_{iteration:03d}.json"
    if out_file.exists():
        print(f"  [skip consolidation] {out_file.name} already exists")
        return

    residual_ids = load_pending_residual_ids(pool)
    if not residual_ids:
        print(f"  consolidation iter={iteration}: no residuals to process")
        return

    # Group residuals by subtype
    by_st = defaultdict(list)
    for nid in residual_ids:
        st = role_of.get(int(nid), role_of.get(nid))
        if not st:
            continue
        a = node_attrs.get(nid) or node_attrs.get(int(nid)) or {}
        by_st[st].append(
            {
                "id": int(nid),
                "name": (a.get("name") or "").strip(),
                "description": (a.get("description") or "").strip(),
                "subtype": st,
            }
        )

    all_new_groups = []
    all_reassignments = []
    per_subtype_stats = {}

    for st, recs in by_st.items():
        if len(recs) < CONSOLIDATION_MIN_GROUP_SIZE:
            per_subtype_stats[st] = {"n_residuals": len(recs), "skipped": "too_few"}
            continue
        print(f"\n  consolidation iter={iteration} subtype={st}: {len(recs)} residuals")
        sentinel = uuid.uuid4().hex[:12]
        prompt = make_consolidation_prompt(st, recs, sentinel)
        print(f"    prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

        client = ClaudeCLI()
        end_marker = f"END_SENTINEL_{sentinel}"
        json_part = None
        for attempt in range(MAX_RETRIES_PER_BATCH + 1):
            print(f"    attempt {attempt + 1}/{MAX_RETRIES_PER_BATCH + 1} ...")
            t0 = time.time()
            try:
                resp = client.messages.create(
                    model="claude-opus-4-7",
                    system=(
                        "You produce STRICT JSON consolidation output. Never preamble, "
                        "never markdown, always emit valid JSON, always end with sentinel."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=SEED_MAX_TOKENS,
                )
                text = resp.content[0].text
                duration = time.time() - t0
                print(f"      returned {len(text)} chars in {duration:.0f}s")
                trimmed = text.strip()
                if trimmed.startswith("{") and trimmed.endswith(end_marker):
                    print("      OK validation")
                    json_part = trimmed[: -len(end_marker)].rstrip()
                    break
            except Exception as e:
                print(f"      FAIL: {type(e).__name__}: {str(e)[:200]}")

        parsed = parse_consolidation_response(json_part)
        if not parsed:
            per_subtype_stats[st] = {"n_residuals": len(recs), "skipped": "llm_failed"}
            continue
        new_groups = parsed.get("new_groups", [])
        reasgn = parsed.get("reassignments", [])
        # Translate node_index -> node_id; attach subtype
        for g in new_groups:
            g["subtype"] = st
            all_new_groups.append(g)
        for r in reasgn:
            ni = r.get("node_index")
            if ni is None or not (0 <= ni < len(recs)):
                continue
            all_reassignments.append(
                {
                    "node_id": recs[ni]["id"],
                    "subtype": st,
                    "new_group_name": r.get("new_group_name"),
                }
            )
        per_subtype_stats[st] = {
            "n_residuals": len(recs),
            "n_new_groups": len(new_groups),
            "n_reassigned": len(reasgn),
            "n_kept_residual": len(recs) - len(reasgn),
        }

    cons_obj = {
        "pool": pool,
        "iteration": iteration,
        "n_residuals_input": len(residual_ids),
        "n_new_groups": len(all_new_groups),
        "n_reassignments": len(all_reassignments),
        "per_subtype": per_subtype_stats,
        "new_groups": all_new_groups,
        "reassignments": all_reassignments,
    }
    atomic_write_json(out_file, cons_obj)
    print(
        f"  consolidation iter={iteration} -> {len(all_new_groups)} new groups, "
        f"{len(all_reassignments)} reassignments saved to {out_file.name}"
    )


def make_review_prompt(pool: str, subtype: str, sample_records, groups, sentinel: str):
    """Pass-A review prompt: assign each sample node to ONE existing group (or 'no_fit')."""
    body = "\n".join(fmt_node(i, r) for i, r in enumerate(sample_records))
    catalog = "\n".join(
        f"  G{i + 1}. {g['name']}\n      desc: {truncate(g['description'], 200)}"
        for i, g in enumerate(groups)
    )
    n = len(sample_records)
    return f"""You are REVIEWING the Pass-A seed catalog for subtype={subtype} by assigning the 200 sample nodes that informed it back to the proposed groups.

PURPOSE: produce a (group_name -> [member node names]) mapping so a human reviewer can see whether the catalog's grouping is mechanistically coherent. This is REVIEW only — no merging, no new groups proposed here.

================================================================
SUBTYPE CONTEXT
================================================================

{SUBTYPE_DEFINITIONS}

================================================================
TARGET FOR THIS CALL: subtype = {subtype}
================================================================

Assign each of the {n} sample nodes to EXACTLY ONE of the {len(groups)} catalog groups, by mechanism-family fit. If a node genuinely fits NO existing group, use `"group_index": null, "decision": "no_fit"` — informational signal that the catalog is missing a class. Do NOT propose new groups.

OUTPUT FORMAT — STRICT:
- Output ONLY one JSON object. No preamble, no markdown.
- Start with `{{`. End with closing `}}` then `END_SENTINEL_{sentinel}` on the same line.
- group_index uses 1-based numbering matching the catalog (G1, G2, ...).

Schema:

{{
  "subtype": "{subtype}",
  "assignments": [
    {{"node_index": 0, "group_index": <1..{len(groups)}>}},
    {{"node_index": 1, "group_index": <1..{len(groups)}>}},
    ...
    {{"node_index": K, "group_index": null, "decision": "no_fit"}}
  ]
}}END_SENTINEL_{sentinel}

CATALOG ({len(groups)} groups):

{catalog}

INPUT NODES ({n} sample nodes; format: `(subtype) name -- description`):

{body}

Now produce the assignment."""


def parse_review_response(text):
    if text is None:
        return None
    t = re.sub(r"^```(?:json)?\s*", "", text.strip())
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except Exception as e:
        print(f"    review parse failed: {e}; trying regex fallback")
        patt = re.compile(
            r'\{"node_index":\s*(\d+),\s*"group_index":\s*(null|\d+)'
            r'(?:,\s*"decision":\s*"([^"]*)")?\s*\}',
            re.DOTALL,
        )
        assignments = []
        for m in patt.finditer(t):
            ni = int(m.group(1))
            gi_raw = m.group(2)
            gi = None if gi_raw == "null" else int(gi_raw)
            d = {"node_index": ni, "group_index": gi}
            if m.group(3):
                d["decision"] = m.group(3)
            assignments.append(d)
        if assignments:
            print(f"    recovered {len(assignments)} assignments via regex")
            return {"assignments": assignments, "_recovered_via_regex": True}
        return None


def run_review(pool: str, role_of: dict, node_attrs: dict):
    """Pass-A REVIEW: assign 200 sample nodes per subtype to proposed groups."""
    print("=" * 80)
    print(f"PASS A — REVIEW (pool={pool}): assign sample nodes to proposed groups")
    print("=" * 80)

    seed_dir = STEP1 / f"phase2_full_vpn_seed_per_subtype_{pool}"
    out_dir = STEP1 / f"phase2_full_vpn_review_{pool}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if pool == "nr":
        subtypes = [
            "problem_analysis",
            "theoretical_insight",
            "design_rationale",
            "implementation_mechanism",
            "validation_evidence",
            "intervention",
        ]
    else:
        subtypes = ["risk"]

    for st in subtypes:
        st_short = SUBTYPE_SHORT.get(st, st)
        seed_file = seed_dir / f"seed_{st_short}.json"
        out_file = out_dir / f"review_{st_short}.json"
        if out_file.exists():
            print(f"\n[skip] {out_file.name} already done")
            continue
        if not seed_file.exists():
            print(f"\nSKIP {st}: seed file missing ({seed_file.name})")
            continue
        seed = json.loads(seed_file.read_text(encoding="utf-8"))
        groups = seed["groups"]
        sample_ids = seed["input_node_ids"]
        # Reconstruct sample records
        sample_records = []
        for nid in sample_ids:
            a = node_attrs.get(nid) or node_attrs.get(int(nid)) or {}
            sample_records.append(
                {
                    "id": int(nid),
                    "name": (a.get("name") or "").strip(),
                    "description": (a.get("description") or "").strip(),
                    "subtype": st,
                }
            )

        print(
            f"\n=== review for subtype={st}: {len(sample_records)} nodes / {len(groups)} groups ==="
        )
        sentinel = uuid.uuid4().hex[:12]
        prompt = make_review_prompt(pool, st, sample_records, groups, sentinel)
        print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

        client = ClaudeCLI()
        end_marker = f"END_SENTINEL_{sentinel}"
        json_part = None
        for attempt in range(MAX_RETRIES_PER_BATCH + 1):
            print(f"  attempt {attempt + 1}/{MAX_RETRIES_PER_BATCH + 1} ...")
            t0 = time.time()
            try:
                resp = client.messages.create(
                    model="claude-opus-4-7",
                    system=(
                        "You produce STRICT JSON assignment output. Never preamble, "
                        "never markdown fences, always emit valid JSON, always end "
                        "with the requested sentinel."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=SEED_MAX_TOKENS,
                )
                text = resp.content[0].text
                duration = time.time() - t0
                print(f"    returned {len(text)} chars in {duration:.0f}s")
                trimmed = text.strip()
                if trimmed.startswith("{") and trimmed.endswith(end_marker):
                    print("    OK validation")
                    json_part = trimmed[: -len(end_marker)].rstrip()
                    break
                else:
                    print(
                        f"    FAIL validation; first/last 100: {repr(trimmed[:100])} ... {repr(trimmed[-100:])}"
                    )
            except Exception as e:
                print(f"    FAIL shim error: {type(e).__name__}: {str(e)[:200]}")

        parsed = parse_review_response(json_part)
        if not parsed or "assignments" not in parsed:
            print(f"  -> {st}: REVIEW FAILED")
            continue

        # Build (group -> [member info]) review structure
        by_group = defaultdict(list)
        no_fit = []
        for a in parsed["assignments"]:
            gi = a.get("group_index")
            ni = a.get("node_index")
            if ni is None or not (0 <= ni < len(sample_records)):
                continue
            rec = sample_records[ni]
            if gi is None:
                no_fit.append({"node_id": rec["id"], "name": rec["name"]})
            elif 1 <= gi <= len(groups):
                gname = groups[gi - 1]["name"]
                by_group[gname].append({"node_id": rec["id"], "name": rec["name"]})

        review_obj = {
            "pool": pool,
            "subtype": st,
            "n_groups": len(groups),
            "n_sample_nodes": len(sample_records),
            "n_assigned": sum(len(v) for v in by_group.values()),
            "n_no_fit": len(no_fit),
            "groups_used": len(by_group),
            "groups_unused": len(groups) - len(by_group),
            "review": [
                {
                    "group_name": g["name"],
                    "group_description": g["description"],
                    "n_members": len(by_group.get(g["name"], [])),
                    "members": by_group.get(g["name"], []),
                }
                for g in groups
            ],
            "no_fit_nodes": no_fit,
            "recovered_via_regex": parsed.get("_recovered_via_regex", False),
        }
        atomic_write_json(out_file, review_obj)
        # Summary line
        sizes = sorted([len(by_group.get(g["name"], [])) for g in groups], reverse=True)
        print(
            f"  -> {st}: {len(by_group)}/{len(groups)} groups got members, "
            f"{len(no_fit)} no_fit, sizes top-5={sizes[:5]} bottom-5={sizes[-5:]}"
        )

    print("\n" + "=" * 80)
    print("REVIEW COMPLETE — files in:", out_dir)
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["smoke", "full", "seed", "review", "consolidate"],
        required=True,
    )
    ap.add_argument("--pool", choices=["nr", "risk", "both"], required=True)
    # --scope controls node-set for Pass B (mode=full):
    #   hdbscan_clustered (default): 1:1 with HDBSCAN scope — for §19.12.5 concentration analysis
    #   all_vpn: includes HDBSCAN-residual nodes — full corpus coverage for §19.0 deliverable 2
    # Idempotent resume means running with all_vpn AFTER hdbscan_clustered automatically
    # processes only the HDBSCAN-residual delta (already-decided nodes are skipped).
    ap.add_argument(
        "--scope", choices=["hdbscan_clustered", "all_vpn"], default="hdbscan_clustered"
    )
    args = ap.parse_args()

    print("loading role_of_rev8_paper.pkl ...")
    with open(STEP1 / "role_of_rev8_paper.pkl", "rb") as f:
        role_of = pickle.load(f)
    print(f"  loaded {len(role_of)} role labels")

    print("loading graph_node_attributes.pkl (3.3GB) ...")
    t0 = time.time()
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  loaded {len(node_attrs)} nodes in {time.time() - t0:.1f}s")

    pools = ["nr", "risk"] if args.pool == "both" else [args.pool]
    for pool in pools:
        if args.mode == "seed":
            run_seed_gen(pool, role_of, node_attrs)
        elif args.mode == "review":
            run_review(pool, role_of, node_attrs)
        elif args.mode == "consolidate":
            # Ad-hoc consolidation pass — auto-assigns iteration number from existing files
            cons_dir = STEP1 / f"phase2_full_vpn_consolidation_{pool}"
            existing = (
                sorted(cons_dir.glob("consolidation_*.json"))
                if cons_dir.exists()
                else []
            )
            iter_num = len(existing) + 1
            run_consolidation(pool, role_of, node_attrs, iter_num)
        else:
            run_pool(pool, args.mode, role_of, node_attrs, scope=args.scope)


if __name__ == "__main__":
    main()
