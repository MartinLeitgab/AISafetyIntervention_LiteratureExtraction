"""
phase2_step4_phase2_seed_taxonomy.py — Phase 2 Task A SEED-ONLY stage

Canonical reproducible script for the rev8 paper Phase 2 LLM-thematic
seed-taxonomy generation. One shim call per pool (risk + non-risk).

Paper intent (Step4_Findings_Report.md §19.0):
The output of this clustering is a structural representation of HOW research
addresses risks in AI safety, used as a downstream-LLM query interface that
bridges the pretraining-data / knowledge-cutoff gap for AI-for-Science work.
Risk-side groups become R-cluster identifiers in `(R_cluster, NR_anchor)`
mechanism-family doublets; NR-side groups become MECHANISM-LEVEL anchors
distinguishing the causal levers research wields (training procedure, policy
lever, evaluation protocol, etc.).

Pool boundaries for Method C (locked 2026-05-08):
- Pool 1 = risk residuals (131 nodes, all included)
- Pool 2 = non-risk residuals (2,095 nodes; sample 250 proportional across
  pa/ti/dr/im/va/intervention for the seed call)

Pipeline architecture (Option A — HDBSCAN cross-check at LLM-input time):
- For each residual node, compute cosine similarity to every HDBSCAN cluster
  centroid in its pool (risk vs nr).
- If max sim ≥ 0.65, attach top-3 nearest cluster candidates (with cluster_id +
  representative names + sim) to the node's prompt entry.
- LLM is asked PER NODE to decide: fold into existing HDBSCAN cluster (rescue),
  assign to a new mechanism-level seed group, or leave as residual-after-seed.
- Risk groups: 15-25 thematic groups (broader granularity acceptable since R is
  paired against NR mechanisms in doublets).
- NR groups: 30-50 mechanism-level groups with worked examples in the prompt
  (training mechanism, policy lever, surveillance mechanism, etc.).

Outputs (suffixed by RISK_VERSION / NR_VERSION below):
- phase2_seed_prompt_{risk,nr}_<version>.txt       — materialized prompt (audit trail)
- phase2_seed_taxonomy_{risk,nr}_<version>_raw.txt — raw LLM response
- phase2_seed_taxonomy_{risk,nr}_<version>.json    — parsed groups + per-node decisions
- phase2_seed_taxonomy_nr_v3_recovered.json        — v3 NR groups regex-recovered after CLI stdout truncation
  (risk locked at v2; NR locked at v3)

Reproducibility: this script is committed under martin/main and intended to be
public when the paper releases. All prompts and data-passing logic live here
verbatim — no external prompt files are required.

Token-cost budget (per CLAUDE.md > 10k rule):
  Risk pool:  ~50k Max plan  (12-13k prompt + ~5k response + ~30k overhead)
  NR pool:    ~70k Max plan  (24-25k prompt + ~5k response + ~30k overhead)
  Total:     ~120k Max plan
"""

import json
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Force UTF-8 stdout so print statements with unicode (≥, ✓, etc.) don't crash
# on Windows cp1252 default. Python 3.7+.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Bump shim subprocess timeout to 30 min (default 900s = 15 min). NR seed call
# with 22.7k-token prompt + 30-50 groups + 250 per-node decisions can exceed
# 15 min; risk pool is fine at default. MUST be set before shim import.
os.environ.setdefault("CLAUDE_CLI_TIMEOUT_SEC", "1800")

SHIM_DIR = Path("C:/Users/malei/0_project_work/0_domain_finder/knowledge_pipeline/src")
sys.path.insert(0, str(SHIM_DIR))
from claude_cli_shim import ClaudeCLI  # noqa: E402

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"

NR_BODY_SUBTYPES = {
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
}
NR_ALL_SUBTYPES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
    "intervention",
]
SUBTYPE_SHORT = {
    "problem_analysis": "pa",
    "theoretical_insight": "ti",
    "design_rationale": "dr",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
    "intervention": "interv",
    "risk": "risk",
}

NEAR_FLOOR_SIM = 0.65  # min cosine sim to surface an HDBSCAN cluster as a candidate
NR_SAMPLE_TARGET = 150  # number of NR-pool residuals to send in the seed call (v3: reduced from 250 for tighter convergence)

# Version suffixes per pool — bump independently when re-running with revised prompts
RISK_VERSION = "v2"  # locked 2026-05-08 — 24 groups, 18% HDBSCAN-rescued, 1 residual
NR_VERSION = "v3"  # 2026-05-08 — abstract-mechanism-class granularity, min-2-members rule, 150-node sample


# ----------------------------------------------------------------------
# Embedding helper — node_attrs stores embedding as FalkorDB string
# '<v1, v2, ...>'. Defensive parser accepts numpy array or string.
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Load all inputs
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Phase 2 SEED-ONLY (v2 — HDBSCAN cross-check + mechanism-family NR prompt)")
    print("=" * 80)

    with open(STEP1 / "phase2_residual_ids_c75m3_subtype.json") as f:
        residual = json.load(f)
    risk_ids = sorted(int(x) for x in residual["risk"])
    nr_ids = sorted(int(x) for x in residual["nr"])
    print(f"residual sizes: risk={len(risk_ids)}, nr={len(nr_ids)}")

    with open(STEP1 / "role_of_rev8_paper.pkl", "rb") as f:
        role_of = pickle.load(f)

    with open(
        STEP1 / "cluster_memberships_rev8_paper_methodA_c75m3_subtype.pkl", "rb"
    ) as f:
        cm_A = pickle.load(f)
    print(f"HDBSCAN clusters loaded: {len(cm_A)} total")

    print("loading graph_node_attributes.pkl (3.3GB) ...")
    t0 = time.time()
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  loaded {len(node_attrs)} nodes in {time.time() - t0:.1f}s")

    def emb_of(nid):
        a = node_attrs.get(nid) or node_attrs.get(int(nid)) or {}
        return parse_emb(a.get("embedding"))

    # --------------------------------------------------------------
    # Compute HDBSCAN cluster centroids + representative names
    # --------------------------------------------------------------
    print("\ncomputing HDBSCAN cluster centroids ...")
    t0 = time.time()
    cluster_info = {}
    for key, members in cm_A.items():
        subtype = key[2]  # 'risk' / 'problem_analysis' / ... / 'intervention'
        cid = key[4]
        full_cid = f"{subtype}_{cid}"
        members = [int(m) for m in members]
        embs = [emb_of(m) for m in members]
        embs = [e for e in embs if e is not None]
        if not embs:
            continue
        centroid = np.mean(embs, axis=0)
        cn = float(np.linalg.norm(centroid))
        if cn > 0:
            centroid = centroid / cn
        sims = [float(np.dot(e, centroid)) for e in embs]
        order = np.argsort(sims)[::-1][:3]
        rep_ids = [members[i] for i in order]
        rep_names = [
            (node_attrs.get(rid) or {}).get("name", "")[:80] for rid in rep_ids
        ]
        pool = "risk" if subtype == "risk" else "nr"
        cluster_info[full_cid] = {
            "pool": pool,
            "subtype": subtype,
            "size": len(members),
            "centroid": centroid,
            "rep_names": rep_names,
            "rep_node_ids": rep_ids,
        }
    print(
        f"  centroids computed for {len(cluster_info)} clusters in {time.time() - t0:.1f}s"
    )

    pool_to_cids = defaultdict(list)
    for cid, info in cluster_info.items():
        pool_to_cids[info["pool"]].append(cid)
    risk_mat = np.stack([cluster_info[c]["centroid"] for c in pool_to_cids["risk"]])
    nr_mat = np.stack([cluster_info[c]["centroid"] for c in pool_to_cids["nr"]])
    print(
        f"  risk pool: {len(pool_to_cids['risk'])} clusters, NR pool: {len(pool_to_cids['nr'])} clusters"
    )

    def top_k_nearest(nid, pool, k=3, min_sim=NEAR_FLOOR_SIM):
        e = emb_of(nid)
        if e is None:
            return []
        if pool == "risk":
            mat, cids = risk_mat, pool_to_cids["risk"]
        else:
            mat, cids = nr_mat, pool_to_cids["nr"]
        sims = mat @ e
        order = np.argsort(sims)[::-1][:k]
        out = []
        for idx in order:
            s = float(sims[idx])
            if s < min_sim:
                break
            out.append(
                {
                    "cid": cids[idx],
                    "sim": s,
                    "rep_names": cluster_info[cids[idx]]["rep_names"],
                    "subtype": cluster_info[cids[idx]]["subtype"],
                }
            )
        return out

    # --------------------------------------------------------------
    # Build node records (with HDBSCAN candidates attached)
    # --------------------------------------------------------------
    def fetch(nid, pool):
        a = node_attrs.get(nid) or node_attrs.get(int(nid)) or {}
        return {
            "id": int(nid),
            "name": (a.get("name") or "").strip(),
            "description": (a.get("description") or "").strip(),
            "concept_category": (a.get("concept_category") or "").strip(),
            "subtype": role_of.get(int(nid), role_of.get(nid, "unknown")),
            "candidates": top_k_nearest(int(nid), pool, k=3, min_sim=NEAR_FLOOR_SIM),
        }

    risk_records = [fetch(nid, "risk") for nid in risk_ids]
    n_risk_cand = sum(1 for r in risk_records if r["candidates"])
    print(
        f"\nRisk: {len(risk_records)} nodes, {n_risk_cand} have ≥1 HDBSCAN candidate ≥ {NEAR_FLOOR_SIM}"
    )

    nr_by_subtype = defaultdict(list)
    for nid in nr_ids:
        rl = role_of.get(int(nid), role_of.get(nid, "unknown"))
        nr_by_subtype[rl].append(int(nid))
    random.seed(42)
    total_nr = sum(len(nr_by_subtype.get(s, [])) for s in NR_ALL_SUBTYPES)
    nr_sample_ids, sample_breakdown = [], {}
    for st in NR_ALL_SUBTYPES:
        pool = nr_by_subtype.get(st, [])
        quota = round(NR_SAMPLE_TARGET * len(pool) / total_nr) if pool else 0
        quota = min(quota, len(pool))
        samp = random.sample(pool, quota) if quota > 0 else []
        nr_sample_ids.extend(samp)
        sample_breakdown[st] = len(samp)
    nr_records = [fetch(nid, "nr") for nid in nr_sample_ids]
    n_nr_cand = sum(1 for r in nr_records if r["candidates"])
    print(
        f"NR sample: {len(nr_records)} nodes, {n_nr_cand} have ≥1 HDBSCAN candidate ≥ {NEAR_FLOOR_SIM}"
    )
    print(f"  NR sample breakdown: {sample_breakdown}")

    # --------------------------------------------------------------
    # Prompt formatters
    # --------------------------------------------------------------
    def truncate(s, n):
        return s if len(s) <= n else s[: n - 1] + "…"

    def fmt_candidates(cands):
        if not cands:
            return ""
        parts = [
            f"{c['cid']} (sim={c['sim']:.2f}: {' / '.join(c['rep_names'][:2])})"
            for c in cands
        ]
        return f"  [HDBSCAN candidates: {' | '.join(parts)}]"

    def fmt_risk_line(i, r):
        cat = r.get("concept_category", "")
        tag = f" [{cat}]" if cat else ""
        return (
            f"{i}. {truncate(r['name'], 100)}{tag} — "
            f"{truncate(r['description'], 250)}{fmt_candidates(r['candidates'])}"
        )

    def fmt_nr_line(i, r):
        st = r["subtype"]
        short = SUBTYPE_SHORT.get(st, st)
        return (
            f"{i}. ({short}) {truncate(r['name'], 100)} — "
            f"{truncate(r['description'], 250)}{fmt_candidates(r['candidates'])}"
        )

    # --------------------------------------------------------------
    # Risk seed prompt (broader thematic granularity, HDBSCAN cross-check)
    # --------------------------------------------------------------
    def make_risk_prompt(records):
        body = "\n".join(fmt_risk_line(i, r) for i, r in enumerate(records))
        return f"""You are an AI safety domain expert producing a thematic taxonomy of RISK CONCEPT residuals from a literature-extracted knowledge graph.

PAPER INTENT:
This corpus was constructed to enable downstream LLMs to bridge the knowledge-cutoff/pretraining-data gap for AI-for-Science work. The clustering output is a structural representation of HOW research addresses risks. Risk-side groups (yours) become R-cluster identifiers in `(R_cluster, NR_anchor)` doublets, paired with mechanism-level non-risk anchors. Risk groups can be coarser/thematic; the precision side is on NR.

DOMAIN: Each input is a node representing a discrete failure mode, harm, or threat scenario in AI systems or related research areas (medical, financial, geopolitical, environmental, etc.) discussed in AI safety literature. Out-of-AI-safety risks are still in scope — they appear in the corpus because the corpus collected them as relevant.

PIPELINE CONTEXT (READ CAREFULLY):
- The full risk-node corpus has 2,464 nodes; 2,333 (94.7%) were already grouped by HDBSCAN-2D clustering at cosine sim >= 0.75 to a cluster centroid. The {len(records)} nodes below are the 5.3% RESIDUAL — they did NOT pass the 0.75 floor.
- Each residual node may have up to 3 HDBSCAN cluster candidates listed (cluster_id + representative names + cosine sim) — those with centroid sim >= {NEAR_FLOOR_SIM:.2f}. A high-sim candidate (sim >= 0.70) likely indicates the node should have been clustered there but missed the strict 0.75 floor.
- Your job is to (a) PER NODE decide whether to fold into an existing HDBSCAN cluster (rescue) or assign to a new seed-taxonomy group, AND (b) propose 15-25 NEW thematic groups for the un-folded residuals.

INPUT: {len(records)} residual risk nodes, indexed 0 to {len(records) - 1}. Format: `name [concept_category] — description  [HDBSCAN candidates: cluster_id (sim=X.XX: rep1 / rep2)]`.

{body}

TASK (TWO-PART):

(1) PER-NODE DECISION — for each input index, output ONE of:
   - `{{"index": N, "decision": "hdbscan", "cluster_id": "risk_42", "confidence": "high"|"medium"}}` — fold into an existing HDBSCAN cluster (only if thematically appropriate; "high" requires strong thematic match; "medium" if the candidate is plausible but not perfect)
   - `{{"index": N, "decision": "seed", "group_name": "<one of your new group names>"}}` — assign to a new seed-taxonomy group
   - `{{"index": N, "decision": "residual"}}` — leave as residual-after-seed (use for ~5% max — genuine misfits)

(2) SEED TAXONOMY — propose 15-25 NEW groups for the nodes you did NOT fold into HDBSCAN clusters.
   - Each group: name (3-8 words, specific not generic — no "Other"/"Misc"/"Miscellaneous"), description (1-2 sentences), representative_indices (2-4 input indices)
   - Group names you create here MUST be referenced verbatim by the `group_name` field in `node_decisions`.

OUTPUT FORMAT: One JSON object. No markdown fences. No commentary. Schema:

{{
  "node_decisions": [
    {{"index": 0, "decision": "hdbscan", "cluster_id": "risk_42", "confidence": "high"}},
    {{"index": 1, "decision": "seed", "group_name": "..."}},
    {{"index": 2, "decision": "residual"}}
  ],
  "groups": [
    {{"name": "...", "description": "...", "representative_indices": [0, 5, 12]}}
  ],
  "notes": "1-2 sentence note on input distribution, fold-in rate, anomalies."
}}

Produce only the JSON object."""

    # --------------------------------------------------------------
    # NR seed prompt v3 (abstract-mechanism-class granularity, hard min-2-members)
    # Lessons from v2: with instance-level granularity the LLM produced 157
    # singleton groups (one per node). v3 forces ABSTRACT MECHANISM CLASS
    # granularity and a hard floor of 2 members per group.
    # --------------------------------------------------------------
    def make_nr_prompt(records):
        body = "\n".join(fmt_nr_line(i, r) for i, r in enumerate(records))
        return f"""You are an AI safety domain expert producing a MECHANISM-FAMILY taxonomy of NON-RISK residuals from a literature-extracted knowledge graph.

PAPER INTENT:
This clustering output is a structural representation of HOW research addresses risks. The downstream paper builds mechanism families as `(R_cluster, NR_anchor)` doublets — your NR groups become NR_anchors. Each NR group must be a MECHANISM-FAMILY, capturing a CLASS of causal lever (not a single instance).

GRANULARITY: ABSTRACT MECHANISM CLASS (NOT instance, NOT topic)

The right granularity is the ABSTRACT MECHANISM CLASS — a category of causal lever that multiple specific implementations share. NOT one node per group, NOT a broad topic.

WORKED EXAMPLES showing the three failure modes and the right level:

EXAMPLE 1 — Diffusion sampling
- Too topical (v1 failure): "Diffusion & generative model sampling"
- Too narrow (v2 failure): "Heun 2nd-order deterministic diffusion sampler" / "DPM-Solver fast diffusion sampler" / "ancestral diffusion sampler" — three groups for what's one mechanism class
- RIGHT (v3 target): "Diffusion-sampling mechanism: deterministic vs stochastic schedule design" — one group covering Heun + DPM-Solver + ancestral

EXAMPLE 2 — RL training
- Too topical: "RL algorithms & sample-efficient learning"
- Too narrow: "Off-policy importance-weighted update" / "KL-regularized policy update" / "Prioritized experience replay" — three singleton groups
- RIGHT: "RL value-update mechanism: stability and credit-assignment design" — one group covering all three

EXAMPLE 3 — Interpretability
- Too topical: "Mechanistic interpretability & circuit analysis"
- Too narrow: "Logit-lens layer-wise probing" / "Sparse-autoencoder feature decomposition" / "Causal-tracing fact localization" / "Filter-concept alignment via Broden" — four singleton groups
- RIGHT: split into ~2 groups
  - "Interpretability mechanism: layer-wise feature-attribution probing" (covers logit-lens + Broden + linear probes)
  - "Interpretability mechanism: causal/structural circuit decomposition" (covers SAE + causal-tracing + activation patching)

EXAMPLE 4 — Policy / governance
- Too topical: "AI governance, licensing & regulatory enforcement"
- Too narrow: "Licensing/revocation regulatory regime" / "Industrial-policy compute control" / "Export-control regime"
- RIGHT (v3 target): split into ~2 groups by lever-type
  - "Policy lever: ex-ante regulatory gating (licensing, registration, certification)"
  - "Policy lever: ex-post enforcement and access restriction (industrial policy, export control)"

A group at the right level: a CLASS of mechanism that multiple papers in the residual share, where each member implements a specific instance of the same CLASS of causal lever. The group name reads as a category, not a paper title.

HARD CONSTRAINTS (MANDATORY — outputs violating these will be rejected):
1. **MIN MEMBERS per group: 2.** Every group in `groups` must have AT LEAST 2 input nodes assigned to it via `node_decisions[].group_name`. **NO SINGLETON GROUPS.** If a node has no mechanistic neighbor in this batch, fold it to the closest existing group or HDBSCAN candidate — DO NOT create a new singleton group.
2. **TARGET GROUP COUNT: 25-45.** Not 100, not 150. Groups should average 3-8 members each.
3. **OUTPUT CONSISTENCY**: every `group_name` referenced in `node_decisions` MUST appear verbatim in the `groups` list. Cross-check before emitting.
4. **NO topical umbrellas, NO singletons.** Group names start with the mechanism class (e.g., "Training mechanism: ...", "Policy lever: ...", "Surveillance mechanism: ..."), then a colon, then the abstract class descriptor.

PIPELINE CONTEXT:
- 250 NR residuals total; 150 sampled here proportionally across pa/ti/dr/im/va/intervention.
- Each residual may have up to 3 HDBSCAN cluster candidates (centroid sim >= {NEAR_FLOOR_SIM:.2f}). cluster_id format `<subtype>_<cluster_id>`.
- Subtype labels (pa/ti/dr/im/va/interv) on input nodes are INFORMATIONAL — DO NOT use subtypes as group boundaries.
- A residual may fold into an HDBSCAN cluster of a DIFFERENT subtype (allowed and welcome).

INPUT: {len(records)} residual non-risk nodes, indexed 0 to {len(records) - 1}. Format: `(subtype) name — description  [HDBSCAN candidates: ...]`.

{body}

TASK (TWO-PART):

(1) PER-NODE DECISION — for each input index 0..{len(records) - 1}, output ONE of:
   - `{{"index": N, "decision": "hdbscan", "cluster_id": "implementation_mechanism_42", "confidence": "high"|"medium"}}`
   - `{{"index": N, "decision": "seed", "group_name": "<verbatim from groups list>"}}`
   - `{{"index": N, "decision": "residual"}}` — for genuine misfits ONLY (~5% max)

(2) MECHANISM-CLASS SEED TAXONOMY — 25-45 groups, each with ≥2 input nodes mapping to it.
   - Each group: name (mechanism-class phrasing per worked examples), description (1-2 sentences naming the causal lever class), representative_indices (2-4 input indices that exemplify it)
   - VERIFY before emitting: count members from your node_decisions per group_name; drop or merge any group with <2 members.

OUTPUT: One JSON object, no fences, no commentary. Schema:

{{
  "node_decisions": [
    {{"index": 0, "decision": "hdbscan", "cluster_id": "implementation_mechanism_15", "confidence": "high"}},
    {{"index": 1, "decision": "seed", "group_name": "..."}}
  ],
  "groups": [
    {{"name": "...", "description": "...", "representative_indices": [0, 5, 12]}}
  ],
  "notes": "1-2 sentence note on input distribution, fold-in rate, granularity choices."
}}

Produce only the JSON object."""

    # --------------------------------------------------------------
    # Shim invocation
    # --------------------------------------------------------------
    def call_shim(prompt, label):
        print(f"\n--- calling shim for {label} pool ---")
        print(f"prompt length: {len(prompt)} chars (~{len(prompt) // 4} tokens)")
        client = ClaudeCLI()
        t0_local = time.time()
        resp = client.messages.create(
            model="claude-opus-4-7",
            system="You produce clean structured JSON taxonomies for AI safety knowledge-graph residuals. Be precise, mechanism-focused where applicable, and reviewer-defensible. Always emit valid JSON with the requested schema.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16384,
        )
        duration = time.time() - t0_local
        text = resp.content[0].text
        print(
            f"  shim returned in {duration:.1f}s, response {len(text)} chars (~{len(text) // 4} tokens)"
        )
        return text, duration

    def parse_json_safe(text):
        t = text.strip()
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
        try:
            return json.loads(t)
        except Exception as e:
            return {"_parse_error": str(e), "_raw_first_500": text[:500]}

    # --------------------------------------------------------------
    # Risk pool (idempotent — skip if output JSON already on disk)
    # --------------------------------------------------------------
    risk_prompt = make_risk_prompt(risk_records)
    (STEP1 / "phase2_seed_prompt_risk_v2.txt").write_text(risk_prompt, encoding="utf-8")
    risk_json_path = STEP1 / "phase2_seed_taxonomy_risk_v2.json"
    if risk_json_path.exists() and os.environ.get("FORCE_RERUN_RISK") != "1":
        print(
            f"\n[idempotent skip] {risk_json_path.name} exists; skipping RISK shim call. "
            f"Set FORCE_RERUN_RISK=1 to override."
        )
        risk_out = json.loads(risk_json_path.read_text(encoding="utf-8"))
        decisions_r = risk_out.get("parsed", {}).get("node_decisions", [])
        n_groups_r = len(risk_out.get("parsed", {}).get("groups", []))
        n_hdbscan_r = sum(1 for d in decisions_r if d.get("decision") == "hdbscan")
        n_seed_r = sum(1 for d in decisions_r if d.get("decision") == "seed")
        n_resid_r = sum(1 for d in decisions_r if d.get("decision") == "residual")
        print(
            f"RISK v2 (cached): {n_groups_r} groups, {len(decisions_r)} decisions "
            f"({n_hdbscan_r} HDBSCAN-rescued, {n_seed_r} seed, {n_resid_r} residual)"
        )
    else:
        risk_raw, risk_dur = call_shim(risk_prompt, "RISK v2")
        risk_parsed = parse_json_safe(risk_raw)
        (STEP1 / "phase2_seed_taxonomy_risk_v2_raw.txt").write_text(
            risk_raw, encoding="utf-8"
        )
        n_groups_r = len(risk_parsed.get("groups", []))
        decisions_r = risk_parsed.get("node_decisions", [])
        n_hdbscan_r = sum(1 for d in decisions_r if d.get("decision") == "hdbscan")
        n_seed_r = sum(1 for d in decisions_r if d.get("decision") == "seed")
        n_resid_r = sum(1 for d in decisions_r if d.get("decision") == "residual")
        risk_out = {
            "pool": "risk",
            "version": "v2_hdbscan_xcheck",
            "n_residual": len(risk_ids),
            "n_input": len(risk_records),
            "n_with_hdbscan_candidates": n_risk_cand,
            "duration_sec": round(risk_dur, 1),
            "summary": {
                "n_groups_proposed": n_groups_r,
                "n_decisions_total": len(decisions_r),
                "n_hdbscan_rescued": n_hdbscan_r,
                "n_seed_assigned": n_seed_r,
                "n_residual_after_seed": n_resid_r,
            },
            "parsed": risk_parsed,
            "input_ids": [r["id"] for r in risk_records],
        }
        risk_json_path.write_text(json.dumps(risk_out, indent=2), encoding="utf-8")
        print(
            f"RISK v2: {n_groups_r} new groups, {len(decisions_r)} decisions "
            f"({n_hdbscan_r} HDBSCAN-rescued, {n_seed_r} seed, {n_resid_r} residual)"
        )

    # --------------------------------------------------------------
    # NR pool (idempotent — skip if output JSON already on disk)
    # --------------------------------------------------------------
    nr_prompt = make_nr_prompt(nr_records)
    (STEP1 / f"phase2_seed_prompt_nr_{NR_VERSION}.txt").write_text(
        nr_prompt, encoding="utf-8"
    )
    nr_json_path = STEP1 / f"phase2_seed_taxonomy_nr_{NR_VERSION}.json"
    if nr_json_path.exists() and os.environ.get("FORCE_RERUN_NR") != "1":
        print(
            f"\n[idempotent skip] {nr_json_path.name} exists; skipping NR shim call. "
            f"Set FORCE_RERUN_NR=1 to override."
        )
        nr_out = json.loads(nr_json_path.read_text(encoding="utf-8"))
        decisions_n = nr_out.get("parsed", {}).get("node_decisions", [])
        n_groups_n = len(nr_out.get("parsed", {}).get("groups", []))
        n_hdbscan_n = sum(1 for d in decisions_n if d.get("decision") == "hdbscan")
        n_seed_n = sum(1 for d in decisions_n if d.get("decision") == "seed")
        n_resid_n = sum(1 for d in decisions_n if d.get("decision") == "residual")
        print(
            f"NR v2 (cached): {n_groups_n} groups, {len(decisions_n)} decisions "
            f"({n_hdbscan_n} HDBSCAN-rescued, {n_seed_n} seed, {n_resid_n} residual)"
        )
        print("\n" + "=" * 80)
        print("SEED-ONLY v2 STAGE DONE")
        print("=" * 80)
        return

    nr_raw, nr_dur = call_shim(nr_prompt, f"NR {NR_VERSION}")
    nr_parsed = parse_json_safe(nr_raw)
    (STEP1 / f"phase2_seed_taxonomy_nr_{NR_VERSION}_raw.txt").write_text(
        nr_raw, encoding="utf-8"
    )
    n_groups_n = len(nr_parsed.get("groups", []))
    decisions_n = nr_parsed.get("node_decisions", [])
    n_hdbscan_n = sum(1 for d in decisions_n if d.get("decision") == "hdbscan")
    n_seed_n = sum(1 for d in decisions_n if d.get("decision") == "seed")
    n_resid_n = sum(1 for d in decisions_n if d.get("decision") == "residual")
    nr_out = {
        "pool": "nr",
        "version": f"{NR_VERSION}_abstract_mechanism_class_hdbscan_xcheck",
        "n_residual": len(nr_ids),
        "n_sampled": len(nr_records),
        "n_with_hdbscan_candidates": n_nr_cand,
        "duration_sec": round(nr_dur, 1),
        "summary": {
            "n_groups_proposed": n_groups_n,
            "n_decisions_total": len(decisions_n),
            "n_hdbscan_rescued": n_hdbscan_n,
            "n_seed_assigned": n_seed_n,
            "n_residual_after_seed": n_resid_n,
        },
        "parsed": nr_parsed,
        "input_ids": [r["id"] for r in nr_records],
        "sample_subtype_breakdown": sample_breakdown,
    }
    nr_json_path.write_text(json.dumps(nr_out, indent=2), encoding="utf-8")
    print(
        f"NR v2: {n_groups_n} new groups, {len(decisions_n)} decisions "
        f"({n_hdbscan_n} HDBSCAN-rescued, {n_seed_n} seed, {n_resid_n} residual)"
    )

    print("\n" + "=" * 80)
    print("SEED-ONLY v2 STAGE DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
