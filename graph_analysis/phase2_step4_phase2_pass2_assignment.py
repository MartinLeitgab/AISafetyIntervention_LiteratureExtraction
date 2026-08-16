"""
phase2_step4_phase2_pass2_assignment.py — Phase 2 Task A PASS-2 ASSIGNMENT

Assigns all 2,095 NR residual nodes to one of:
  (a) an existing HDBSCAN cluster (rescue) — `(subtype, cluster_id)` pair
  (b) one of the 33 v3 mechanism-class seed groups (LLM Method C)
  (c) "residual" — last-resort, ~5% max

Pass-2 follows the seed-taxonomy stage (phase2_step4_phase2_seed_taxonomy.py):
  - Risk pool: already complete (24 v2 LLM groups + 23 HDBSCAN rescues + 1 residual,
    locked from v2 seed call). No risk Pass-2 needed.
  - NR pool: 2,095 residuals to assign against (3,396 HDBSCAN clusters ∪ 33 v3 LLM groups).

Anti-truncation safeguards (per user instruction "no truncation, no timeout"):
  1. Smaller batches (80 nodes/batch → 26 batches sequential) to keep response < 4k tokens.
  2. SENTINEL validation: prompt asks LLM to end response with `}END_SENTINEL_<UUID>`. If
     the response doesn't end with the expected sentinel, treat as truncated → retry.
  3. START-MARKER validation: response must start with `{` (no preamble). If not,
     truncated/preamble-leaked → retry.
  4. 60-min timeout per call (CLAUDE_CLI_TIMEOUT_SEC=3600) — no time cutoff.
  5. Per-batch idempotency: each batch saves its own JSON; resume by skipping existing
     batch outputs. Recoverable from any partial state.
  6. JSON parse + regex fallback if final parse fails after retries.

Outputs:
  - phase2_pass2_batch_NN.json                            (per-batch decisions)
  - phase2_pass2_decisions_all.json                       (merged after all batches)
  - cluster_memberships_rev8_paper_methodC_c75m3_v3.pkl   (group_name → [node_ids])
  - phase2_pass2_summary.json                             (final counts, stats)

Token-cost budget (per CLAUDE.md > 10k rule):
  Per batch: ~22k prompt + ~3k response + ~30k overhead = ~55k Max plan
  26 batches × 55k = ~1.43M Max plan
  +retry budget (1-3 batches at most) = +150k
  Total estimated: ~1.45-1.6M Max plan
"""

import json
import os
import pickle
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

# 60-min subprocess timeout — per user instruction "no time cutoff"
os.environ.setdefault("CLAUDE_CLI_TIMEOUT_SEC", "3600")

SHIM_DIR = Path("C:/Users/malei/0_project_work/0_domain_finder/knowledge_pipeline/src")
sys.path.insert(0, str(SHIM_DIR))
from claude_cli_shim import ClaudeCLI  # noqa: E402

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
PASS2_DIR = STEP1 / "phase2_pass2_batches"
PASS2_DIR.mkdir(exist_ok=True)

# ---- Configuration ----
NR_BATCH_SIZE = 80
NEAR_FLOOR_SIM = 0.65
MAX_RETRIES_PER_BATCH = 2
NR_VERSION = "v3"


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
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    print("=" * 80)
    print("Phase 2 Task A PASS-2 — NR residual assignment (2,095 nodes)")
    print("=" * 80)

    # ---- Load inputs ----
    with open(STEP1 / "phase2_residual_ids_c75m3_subtype.json") as f:
        residual = json.load(f)
    nr_ids = sorted(int(x) for x in residual["nr"])
    print(f"NR residual count: {len(nr_ids)}")

    with open(STEP1 / "role_of_rev8_paper.pkl", "rb") as f:
        role_of = pickle.load(f)

    with open(
        STEP1 / "cluster_memberships_rev8_paper_methodA_c75m3_subtype.pkl", "rb"
    ) as f:
        cm_A = pickle.load(f)
    print(f"HDBSCAN clusters: {len(cm_A)}")

    with open(
        STEP1 / "phase2_seed_taxonomy_nr_v3_recovered.json", encoding="utf-8"
    ) as f:
        v3_seed = json.load(f)
    seed_groups = v3_seed["parsed"]["groups"]
    print(f"v3 seed groups: {len(seed_groups)}")

    print("loading graph_node_attributes.pkl (3.3GB) ...")
    t0 = time.time()
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  loaded {len(node_attrs)} nodes in {time.time() - t0:.1f}s")

    def emb_of(nid):
        a = node_attrs.get(nid) or node_attrs.get(int(nid)) or {}
        return parse_emb(a.get("embedding"))

    # ---- HDBSCAN cluster centroids (NR pool only) ----
    print("\ncomputing NR HDBSCAN cluster centroids ...")
    t0 = time.time()
    cluster_info = {}
    for key, members in cm_A.items():
        subtype = key[2]
        if subtype == "risk":  # NR pool only for Pass-2
            continue
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
        cluster_info[full_cid] = {
            "subtype": subtype,
            "size": len(members),
            "centroid": centroid,
            "rep_names": rep_names,
        }
    print(
        f"  centroids computed for {len(cluster_info)} NR clusters in {time.time() - t0:.1f}s"
    )

    nr_cids = list(cluster_info.keys())
    nr_centroid_matrix = np.stack([cluster_info[c]["centroid"] for c in nr_cids])

    def top_k_nearest(nid, k=3, min_sim=NEAR_FLOOR_SIM):
        e = emb_of(nid)
        if e is None:
            return []
        sims = nr_centroid_matrix @ e
        order = np.argsort(sims)[::-1][:k]
        out = []
        for idx in order:
            s = float(sims[idx])
            if s < min_sim:
                break
            out.append(
                {
                    "cid": nr_cids[idx],
                    "sim": s,
                    "rep_names": cluster_info[nr_cids[idx]]["rep_names"],
                    "subtype": cluster_info[nr_cids[idx]]["subtype"],
                }
            )
        return out

    # ---- Build per-node records once ----
    SUBTYPE_SHORT = {
        "problem_analysis": "pa",
        "theoretical_insight": "ti",
        "design_rationale": "dr",
        "implementation_mechanism": "im",
        "validation_evidence": "va",
        "intervention": "interv",
    }

    print("\nbuilding per-node records with HDBSCAN candidates ...")
    t0 = time.time()
    node_records = []
    for nid in nr_ids:
        a = node_attrs.get(nid) or node_attrs.get(int(nid)) or {}
        node_records.append(
            {
                "id": int(nid),
                "name": (a.get("name") or "").strip(),
                "description": (a.get("description") or "").strip(),
                "subtype": role_of.get(int(nid), role_of.get(nid, "unknown")),
                "candidates": top_k_nearest(int(nid), k=3, min_sim=NEAR_FLOOR_SIM),
            }
        )
    n_with_cands = sum(1 for r in node_records if r["candidates"])
    print(f"  built {len(node_records)} records in {time.time() - t0:.1f}s")
    print(
        f"  {n_with_cands} ({100 * n_with_cands / len(node_records):.1f}%) have ≥1 HDBSCAN candidate ≥ {NEAR_FLOOR_SIM}"
    )

    # ---- Build seed-group catalog block (constant across batches) ----
    seed_catalog_lines = []
    for i, g in enumerate(seed_groups):
        seed_catalog_lines.append(
            f"  G{i + 1}. {g['name']}\n      desc: {truncate(g['description'], 200)}"
        )
    seed_catalog = "\n".join(seed_catalog_lines)

    # ---- Per-batch prompt builder ----
    def fmt_node(i, r):
        st = SUBTYPE_SHORT.get(r["subtype"], r["subtype"])
        cands = r["candidates"]
        cand_str = ""
        if cands:
            parts = [
                f"{c['cid']} (sim={c['sim']:.2f}: {' / '.join(c['rep_names'][:2])})"
                for c in cands
            ]
            cand_str = f"  [HDBSCAN candidates: {' | '.join(parts)}]"
        return (
            f"{i}. ({st}) {truncate(r['name'], 100)} — "
            f"{truncate(r['description'], 250)}{cand_str}"
        )

    def make_prompt(records, sentinel):
        body = "\n".join(fmt_node(i, r) for i, r in enumerate(records))
        return f"""You are doing PASS-2 of a clustering pipeline for AI safety literature mechanism-family extraction.

PHASE 1 (HDBSCAN-2D per-subtype cosine ≥ 0.75 floor) clustered 88% of the NR corpus into 3,396 geometric clusters.

PHASE 2 SEED produced {len(seed_groups)} mechanism-class groups for the long-tail residual. Your task: assign each input residual node to ONE of these three options.

OUTPUT FORMAT — STRICT (validation will reject malformed responses):
- Output ONLY one JSON object. No preamble, no markdown fences, no commentary.
- Start your output with the character `{{`.
- After the closing `}}`, append the literal sentinel `END_SENTINEL_{sentinel}` on the same line.

DECISION OPTIONS PER NODE (pick exactly one):
- `{{"index": N, "decision": "hdbscan", "cluster_id": "<full_cid>", "confidence": "high"|"medium"}}` — fold into an existing HDBSCAN cluster (use a `cluster_id` from the node's HDBSCAN candidates list, or any other cluster_id you judge appropriate). Use "high" only when the candidate is clearly mechanistically right; "medium" if plausible but not certain.
- `{{"index": N, "decision": "seed", "group_name": "<verbatim group name from catalog below>"}}` — fold into one of the {len(seed_groups)} mechanism-class seed groups. The group_name MUST match a name in the catalog verbatim.
- `{{"index": N, "decision": "residual"}}` — last resort for genuine misfits (~5% max). Use sparingly.

GUIDANCE:
- Subtype labels (pa/ti/dr/im/va/interv) on input nodes are INFORMATIONAL — DO NOT use as group boundaries. A node may legitimately belong to an HDBSCAN cluster of a DIFFERENT subtype OR to a seed group whose typical members have different subtypes.
- Mechanism family takes priority over surface keyword similarity. If a node is mechanistically the same lever as members of seed group G, fold to G even if the node's specific instance differs from G's representatives.
- HDBSCAN candidates with high sim (≥ 0.70) are strong rescue signals — prefer rescue if mechanistically right.

SEED GROUP CATALOG ({len(seed_groups)} mechanism-class groups):

{seed_catalog}

INPUT NODES ({len(records)} residuals indexed 0 to {len(records) - 1}; format: `(subtype) name — description  [HDBSCAN candidates: ...]`):

{body}

OUTPUT — single JSON object, no preamble, end with `END_SENTINEL_{sentinel}`:

{{
  "node_decisions": [
    {{"index": 0, "decision": "...", ...}},
    {{"index": 1, "decision": "...", ...}}
  ]
}}END_SENTINEL_{sentinel}

Now produce the response."""

    def call_with_validation(prompt, sentinel, label):
        """Call shim; validate sentinel + start-marker; retry on failure."""
        client = ClaudeCLI()
        end_marker = f"END_SENTINEL_{sentinel}"
        for attempt in range(MAX_RETRIES_PER_BATCH + 1):
            print(f"  [{label}] attempt {attempt + 1}/{MAX_RETRIES_PER_BATCH + 1} ...")
            t0_local = time.time()
            try:
                resp = client.messages.create(
                    model="claude-opus-4-7",
                    system="You produce STRICT JSON output for a mechanism-family clustering pipeline. Never preamble, never use markdown fences, always emit valid JSON, always end your output with the requested sentinel.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=16384,
                )
                text = resp.content[0].text
                duration = time.time() - t0_local
                print(f"    returned {len(text)} chars in {duration:.0f}s")
                # Validate
                trimmed = text.strip()
                ok_start = trimmed.startswith("{")
                ok_end = trimmed.endswith(end_marker)
                if ok_start and ok_end:
                    print("    ✓ start-marker + sentinel both present")
                    json_part = trimmed[: -len(end_marker)].rstrip()
                    return json_part, duration, attempt + 1
                else:
                    print(f"    ✗ validation failed (start={ok_start}, end={ok_end})")
                    print(f"      first 100 chars: {repr(trimmed[:100])}")
                    print(f"      last 100 chars: {repr(trimmed[-100:])}")
            except Exception as e:
                print(f"    ✗ shim error: {type(e).__name__}: {str(e)[:200]}")
                duration = time.time() - t0_local
        return None, duration, MAX_RETRIES_PER_BATCH + 1

    def parse_with_fallback(text, batch_size):
        """Parse the JSON object's node_decisions array; regex fallback if parse fails."""
        if text is None:
            return None
        t = re.sub(r"^```(?:json)?\s*", "", text.strip())
        t = re.sub(r"\s*```\s*$", "", t)
        try:
            return json.loads(t)
        except Exception as e:
            print(f"    parse failed: {e}; trying regex fallback")
            # Extract individual decision objects via regex
            patt = re.compile(
                r'\{"index":\s*(\d+),\s*"decision":\s*"(hdbscan|seed|residual)"'
                r'(?:,\s*"cluster_id":\s*"([^"]*)")?'
                r'(?:,\s*"group_name":\s*"((?:[^"\\]|\\.)*)")?'
                r'(?:,\s*"confidence":\s*"([^"]*)")?\}',
                re.DOTALL,
            )
            decisions = []
            for m in patt.finditer(t):
                d = {"index": int(m.group(1)), "decision": m.group(2)}
                if m.group(3):
                    d["cluster_id"] = m.group(3)
                if m.group(4):
                    d["group_name"] = m.group(4)
                if m.group(5):
                    d["confidence"] = m.group(5)
                decisions.append(d)
            if decisions:
                print(
                    f"    regex fallback recovered {len(decisions)} decisions of {batch_size}"
                )
                return {"node_decisions": decisions, "_recovered_via_regex": True}
            return None

    # ---- Run batches ----
    n_batches = (len(node_records) + NR_BATCH_SIZE - 1) // NR_BATCH_SIZE
    print(f"\ntotal batches: {n_batches} ({NR_BATCH_SIZE} nodes/batch)")
    all_decisions = []
    failed_batches = []

    for batch_idx in range(n_batches):
        start = batch_idx * NR_BATCH_SIZE
        end = min(start + NR_BATCH_SIZE, len(node_records))
        batch = node_records[start:end]
        batch_file = PASS2_DIR / f"phase2_pass2_batch_{batch_idx:02d}.json"

        # Idempotency: skip if batch output already exists
        if batch_file.exists():
            print(
                f"\n[idempotent skip] batch {batch_idx:02d} ({start}-{end - 1}) already done, loading"
            )
            saved = json.loads(batch_file.read_text(encoding="utf-8"))
            all_decisions.extend(saved.get("decisions", []))
            continue

        print(
            f"\n=== batch {batch_idx:02d}/{n_batches - 1}: nodes {start}-{end - 1} ({len(batch)} items) ==="
        )
        sentinel = uuid.uuid4().hex[:12]
        prompt = make_prompt(batch, sentinel)
        print(f"  prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)")

        json_part, duration, attempts = call_with_validation(
            prompt, sentinel, f"batch_{batch_idx:02d}"
        )
        parsed = parse_with_fallback(json_part, len(batch)) if json_part else None

        if not parsed or "node_decisions" not in parsed:
            print(
                f"  ✗ batch {batch_idx:02d} FAILED after {attempts} attempts; logging to retry list"
            )
            failed_batches.append(batch_idx)
            # Save failure marker
            batch_file.write_text(
                json.dumps(
                    {
                        "batch_idx": batch_idx,
                        "status": "failed",
                        "node_indices_local": list(range(len(batch))),
                        "node_ids_global": [r["id"] for r in batch],
                        "attempts": attempts,
                        "duration_sec": round(duration, 1),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue

        # Translate local index → global node_id
        decisions = parsed["node_decisions"]
        translated = []
        for d in decisions:
            local_i = d.get("index")
            if local_i is None or not (0 <= local_i < len(batch)):
                continue
            translated.append(
                {
                    "node_id": batch[local_i]["id"],
                    "subtype": batch[local_i]["subtype"],
                    "decision": d.get("decision"),
                    "cluster_id": d.get("cluster_id"),
                    "group_name": d.get("group_name"),
                    "confidence": d.get("confidence"),
                }
            )

        # Save batch output
        batch_out = {
            "batch_idx": batch_idx,
            "status": "ok",
            "n_input": len(batch),
            "n_decisions": len(translated),
            "duration_sec": round(duration, 1),
            "attempts": attempts,
            "recovered_via_regex": parsed.get("_recovered_via_regex", False),
            "decisions": translated,
        }
        batch_file.write_text(json.dumps(batch_out, indent=2), encoding="utf-8")
        all_decisions.extend(translated)
        n_hd = sum(1 for d in translated if d["decision"] == "hdbscan")
        n_seed = sum(1 for d in translated if d["decision"] == "seed")
        n_res = sum(1 for d in translated if d["decision"] == "residual")
        print(
            f"  ✓ batch {batch_idx:02d}: {len(translated)} decisions ({n_hd} hdbscan, {n_seed} seed, {n_res} residual)"
        )

    # ---- Final merge ----
    print("\n" + "=" * 80)
    print("ALL BATCHES DONE — merging")
    print("=" * 80)
    print(f"failed batches: {failed_batches}")
    print(f"total decisions: {len(all_decisions)}")

    # Save combined decisions
    combined = {
        "version": f"pass2_{NR_VERSION}",
        "n_residual_total": len(nr_ids),
        "n_decisions": len(all_decisions),
        "n_failed_batches": len(failed_batches),
        "failed_batch_indices": failed_batches,
        "summary": {
            "n_hdbscan_rescued": sum(
                1 for d in all_decisions if d["decision"] == "hdbscan"
            ),
            "n_seed_assigned": sum(1 for d in all_decisions if d["decision"] == "seed"),
            "n_residual": sum(1 for d in all_decisions if d["decision"] == "residual"),
        },
        "decisions": all_decisions,
    }
    (STEP1 / "phase2_pass2_decisions_all.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    # Build cluster_memberships_methodC PKL: group_name → list of node_ids
    method_c = defaultdict(list)
    for d in all_decisions:
        if d["decision"] == "seed" and d.get("group_name"):
            key = ("rev8_paper", "llm_pass2", "nr", "methodC", d["group_name"])
            method_c[key].append(d["node_id"])

    method_c_path = (
        STEP1 / f"cluster_memberships_rev8_paper_methodC_c75m3_{NR_VERSION}.pkl"
    )
    with open(method_c_path, "wb") as f:
        pickle.dump(dict(method_c), f)
    print(f"\nsaved {method_c_path.name}")
    print(
        f"  {len(method_c)} LLM groups with {sum(len(v) for v in method_c.values())} total members"
    )

    # Save summary
    group_sizes = sorted(
        [(name[4], len(members)) for name, members in method_c.items()],
        key=lambda x: -x[1],
    )
    summary = {
        **combined["summary"],
        "n_failed_batches": len(failed_batches),
        "failed_batch_indices": failed_batches,
        "n_distinct_hdbscan_rescue_clusters": len(
            set(
                d.get("cluster_id") for d in all_decisions if d["decision"] == "hdbscan"
            )
        ),
        "n_distinct_seed_groups_used": len(method_c),
        "top_seed_groups_by_size": group_sizes[:15],
    }
    (STEP1 / "phase2_pass2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\nFINAL SUMMARY:")
    for k, v in summary.items():
        if k != "top_seed_groups_by_size":
            print(f"  {k}: {v}")
    print("  top seed groups by member count:")
    for name, n in group_sizes[:10]:
        print(f"    [{n}] {name}")

    print("\n" + "=" * 80)
    print("PASS-2 ASSIGNMENT DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
