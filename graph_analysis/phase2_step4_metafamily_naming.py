"""
phase2_step4_metafamily_naming.py

Runs causal LLM naming on all 32 PathbuildB meta-families, then cross-checks
consistency with the already-named top-40 individual B-families (v3).

Outputs:
  step5_naming/pathbuildB_metafamily_names_llm.csv   — 32 rows, named meta-families
  step5_naming/pathbuildB_metafamily_consistency.csv — cross-check of v3 B-family
                                                        names vs parent meta-family name
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
client = OpenAI(api_key=api_key)
MODEL = "gpt-4o-mini"

BASE = Path("graph_analysis/phase2_results/step4_finalanalysis")
NAMING_DIR = Path("graph_analysis/phase2_results/step5_naming")
NAMING_DIR.mkdir(exist_ok=True)

# ── Load inputs ────────────────────────────────────────────────────────────────

summ = pd.read_csv(
    BASE / "step4_cluster_tables/pathbuildB_metafamily_summary_consim1.csv"
)
mf_all = pd.read_csv(BASE / "step4_cluster_tables/pathbuildB_metafamilies_consim1.csv")
v3 = pd.read_csv(NAMING_DIR / "pathbuildB_chain_names_llm_v3.csv")
meta_trips = pd.read_csv(BASE / "step4_connectivity/ri_meta_triplets_consim1.csv")
body_reps = pd.read_csv(
    BASE / "step4_cluster_tables/bodysubtype_cluster_representatives.csv"
)

# Decode cluster ID → name from body representatives
# columns expected: cluster_id (or subtype_cluster), top_node_name (or similar)
print("body_reps columns:", body_reps.columns.tolist())
print(body_reps.head(3).to_string())

# Build cluster_id → label map
# prefix_key = "de:15", rep_name = the readable label
body_map = dict(
    zip(body_reps["prefix_key"].astype(str), body_reps["rep_name"].astype(str))
)
print(f"Loaded {len(body_map)} body cluster labels")

# v3 B-family names: cluster_id (0-39) → final_name
v3_map = dict(zip(v3["cluster_id"].astype(int), v3["final_name"].astype(str)))

# meta_family_id → list of v3-named members
mf_all["in_v3"] = mf_all["family_id"].isin(v3_map.keys())
v3_members_by_meta = (
    mf_all[mf_all["in_v3"]].groupby("meta_family_id")["family_id"].apply(list).to_dict()
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def decode_signature(sig_str):
    """Decode 'de:15 & im:4 & pr:6' → readable cluster descriptions using prefix_key map."""
    if not sig_str or pd.isna(sig_str):
        return []
    parts = [p.strip() for p in str(sig_str).split("&")]
    decoded = []
    for part in parts:
        # part is like "de:15" — look it up directly in body_map keyed by prefix_key
        label = body_map.get(part, part)
        decoded.append(label)
    return decoded


def get_top_ri_for_meta(meta_id, n=3):
    """Get top N risk→intervention pairs for a meta-family."""
    sub = meta_trips[meta_trips["meta_family_id"] == meta_id].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("n_triplet_paths", ascending=False).head(n)
    pairs = []
    for _, r in sub.iterrows():
        pairs.append(
            f"  [{r['risk_meta_name']}] → [{r['interv_meta_name']}]  ({int(r['n_triplet_paths'])} paths)"
        )
    return pairs


def call_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=400,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  LLM attempt {attempt + 1} failed: {e}")
            time.sleep(2**attempt)
    return {}


CAUSAL_PROMPT = """\
You are naming a PathbuildB meta chain-family in an AI safety knowledge graph.

CONTEXT:
A meta chain-family groups {n_families} individual reasoning-path families ({n_paths} total paths)
that share a common core mechanism connecting risk clusters to intervention clusters.

Top risk → intervention pairs this meta-family bridges:
{ri_context}

Core body concept clusters (the shared mechanism):
{decoded_text}

Already-named individual B-families within this meta-family (from a prior naming pass):
{v3_members}

Current name (likely WRONG — uses "via"/"through" framing or is non-causal): "{old_name}"

TASK:
Name the shared causal mechanism that explains WHY the interventions mitigate the risks
via this chain family.

REQUIREMENT: Your name must complete this sentence naturally and truthfully:
  "The reason why [the intervention] mitigates [the risk] is [YOUR NAME]"

Example of a good name: "reward signal alignment with human preferences"
→ "The reason why RLHF fine-tuning mitigates deceptive alignment is reward signal alignment with human preferences" ✓

RULES:
1. Do NOT start with "via", "through", "by means of", or any preposition
2. Do NOT restate the intervention or risk name
3. DO describe the intermediate mechanism — what it IS as a causal concept
4. 4–10 words, a noun phrase describing the mechanism
5. Must write a specific test sentence using real risk/intervention names from context above
6. Confirm the sentence is coherent

Respond as valid JSON only:
{{
  "name": "4-10 word causal mechanism noun phrase",
  "description": "one sentence explaining the mechanism",
  "test_sentence": "The reason why [specific intervention from context] mitigates [specific risk from context] is [your name]",
  "test_sentence_ok": true/false,
  "test_sentence_reasoning": "brief explanation",
  "judge_confidence": "high/medium/low"
}}
"""

JUDGE_PROMPT = """\
You are reviewing a proposed meta chain-family name in an AI safety knowledge graph.

Proposed name: "{name}"
Test sentence: "{test_sentence}"
Author confidence: {test_ok}

Core body concepts: {decoded_text}

Top R→I pairs bridged: {ri_context}

Already-named individual B-families in this meta-family: {v3_members}

Review:
1. Does the name describe a causal mechanism (not a preposition phrase starting with via/through)? (starts_via_through: true/false)
2. Is the name consistent with the individual B-family names listed above? (consistent_with_v3: true/false)
3. If inconsistent, explain why and suggest a better name (suggested_revision: "..." or null)
4. Overall confidence (high/medium/low)

Respond as valid JSON only:
{{
  "starts_via_through": true/false,
  "consistent_with_v3": true/false,
  "consistency_note": "brief note on consistency",
  "suggested_revision": "..." or null,
  "judge_confidence": "high/medium/low"
}}
"""

# ── Main naming loop ──────────────────────────────────────────────────────────

results = []

for _, row in summ.iterrows():
    mf_id = int(row["meta_family_id"])
    old_name = str(row["dominant_family_name"])
    n_families = int(row["n_families"])
    n_paths = int(row["n_paths_total"])
    core_sig = str(row.get("core_components", ""))

    print(
        f"\n── Meta-family {mf_id} | {n_families} families | {n_paths} paths | old: {old_name[:60]}"
    )

    decoded = decode_signature(core_sig)
    decoded_text = "\n".join(decoded) if decoded else f"(signature: {core_sig})"
    ri_pairs = get_top_ri_for_meta(mf_id)
    ri_context = "\n".join(ri_pairs) if ri_pairs else "(no triplet data)"

    v3_fam_ids = v3_members_by_meta.get(mf_id, [])
    v3_member_lines = [f"  B{fid}: {v3_map[fid]}" for fid in sorted(v3_fam_ids)]
    v3_members_str = (
        "\n".join(v3_member_lines) if v3_member_lines else "(none in top-40)"
    )

    # Pass 1 — causal naming
    p1 = call_llm(
        CAUSAL_PROMPT.format(
            n_families=n_families,
            n_paths=n_paths,
            ri_context=ri_context,
            decoded_text=decoded_text,
            v3_members=v3_members_str,
            old_name=old_name,
        )
    )

    name = p1.get("name", old_name)
    description = p1.get("description", "")
    test_sentence = p1.get("test_sentence", "")
    test_ok = p1.get("test_sentence_ok", True)
    test_reasoning = p1.get("test_sentence_reasoning", "")
    p1_confidence = p1.get("judge_confidence", "")

    print(f"  → P1 name: {name}")
    print(f"  → Test:    {test_sentence[:100]}")

    # Pass 2 — judge review
    p2 = call_llm(
        JUDGE_PROMPT.format(
            name=name,
            test_sentence=test_sentence,
            test_ok=test_ok,
            decoded_text=decoded_text[:300],
            ri_context=ri_context[:300],
            v3_members=v3_members_str[:400],
        )
    )

    starts_via = p2.get("starts_via_through", False)
    consistent_v3 = p2.get("consistent_with_v3", True)
    consistency_note = p2.get("consistency_note", "")
    suggested = p2.get("suggested_revision", None)
    judge_conf = p2.get("judge_confidence", "")

    final_name = suggested if suggested and (starts_via or not consistent_v3) else name
    print(
        f"  → Judge: via={starts_via}, v3_consistent={consistent_v3}, conf={judge_conf}"
    )
    if suggested:
        print(f"  → Suggested revision: {suggested}")
    print(f"  → Final: {final_name}")

    results.append(
        {
            "meta_family_id": mf_id,
            "n_families": n_families,
            "n_paths_total": n_paths,
            "old_name": old_name,
            "core_components": core_sig,
            "llm_name": name,
            "description": description,
            "test_sentence": test_sentence,
            "test_sentence_ok": test_ok,
            "test_sentence_reasoning": test_reasoning,
            "p1_confidence": p1_confidence,
            "judge_starts_via_through": starts_via,
            "judge_consistent_with_v3": consistent_v3,
            "consistency_note": consistency_note,
            "suggested_revision": suggested,
            "judge_confidence": judge_conf,
            "final_name": final_name,
            "v3_members_in_meta": "; ".join(
                [f"B{fid}:{v3_map[fid]}" for fid in sorted(v3_fam_ids)]
            ),
        }
    )

    time.sleep(0.3)

out_df = pd.DataFrame(results)
out_path = NAMING_DIR / "pathbuildB_metafamily_names_llm.csv"
out_df.to_csv(out_path, index=False)
print(f"\n✅ Written: {out_path}")

# ── Consistency cross-check table ─────────────────────────────────────────────

name_map = dict(zip(out_df["meta_family_id"], out_df["final_name"]))
rows_check = []
for fid, meta_id in zip(mf_all["family_id"], mf_all["meta_family_id"]):
    if int(fid) not in v3_map:
        continue
    rows_check.append(
        {
            "b_family_id": int(fid),
            "b_family_v3_name": v3_map[int(fid)],
            "meta_family_id": int(meta_id),
            "meta_family_final_name": name_map.get(int(meta_id), ""),
        }
    )

check_df = pd.DataFrame(rows_check).drop_duplicates()
check_path = NAMING_DIR / "pathbuildB_metafamily_consistency.csv"
check_df.to_csv(check_path, index=False)
print(f"✅ Written: {check_path}")

# Print summary
print("\n═══ FINAL META-FAMILY NAMES ═══")
for _, r in out_df.sort_values("n_paths_total", ascending=False).iterrows():
    flag = (
        "⚠️ "
        if r["judge_starts_via_through"] or not r["judge_consistent_with_v3"]
        else "✅"
    )
    print(
        f"{flag} MF{r['meta_family_id']:2d} ({r['n_paths_total']:6d} paths): {r['final_name']}"
    )

print("\n═══ CONSISTENCY CHECK (v3 B-families vs meta-family) ═══")
for _, r in check_df.sort_values("meta_family_id").iterrows():
    print(f"  B{r['b_family_id']:3d} [{r['b_family_v3_name'][:50]}]")
    print(f"       ↳ MF{r['meta_family_id']:2d} [{r['meta_family_final_name'][:50]}]")
