"""
phase2_step4_E4_frozenset_group_naming.py  [rev7]

2-pass gpt-4.1-mini naming of 20 frozenset groups.

Context fed to LLM per group:
  1. Centroid decoded body concept names (top-5)
  2. Closest-3 frozensets decoded (most representative)
  3. Farthest-3 frozensets decoded (borderline — where group edges are)
  4. Top-3 R->I pairs this group bridges (from ri_triplets_consim1.csv)

Causal framing: name must complete
  "The reason why [intervention] mitigates [risk] is [NAME]"

Inputs:
  step4_cluster_tables/frozenset_groups_consim1.csv
  step4_cluster_tables/frozenset_group_memberships_consim1.csv
  step4_connectivity/ri_triplets_consim1.csv
  step5_naming/risk_cluster_names_llm_v2.csv
  step5_naming/intervention_cluster_names_llm_v2.csv

Output (NEW):
  step5_naming/frozenset_group_names_llm.csv
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
client = OpenAI(api_key=api_key)
MODEL = "gpt-5.5"

RESULTS_DIR = ROOT / "phase2_results"
STEP4_DIR = RESULTS_DIR / "step4_finalanalysis"
TABLES_DIR = STEP4_DIR / "step4_cluster_tables"
CONN_DIR = STEP4_DIR / "step4_connectivity"
NAMING_DIR = RESULTS_DIR / "step5_naming"

# ── Load inputs ───────────────────────────────────────────────────────────────

groups = pd.read_csv(TABLES_DIR / "frozenset_groups_consim1.csv")
memberships = pd.read_csv(TABLES_DIR / "frozenset_group_memberships_consim1.csv")
ri_trips = pd.read_csv(CONN_DIR / "ri_triplets_consim1.csv")

risk_df = pd.read_csv(NAMING_DIR / "risk_cluster_names_llm_v2.csv")
interv_df = pd.read_csv(NAMING_DIR / "intervention_cluster_names_llm_v2.csv")
risk_col = "final_name" if "final_name" in risk_df.columns else "llm_name"
interv_col = "final_name" if "final_name" in interv_df.columns else "llm_name"
risk_names = dict(zip(risk_df["cluster_id"].astype(str), risk_df[risk_col]))
interv_names = dict(zip(interv_df["cluster_id"].astype(str), interv_df[interv_col]))

# group_id -> set of family_ids
group_families = memberships.groupby("group_id")["family_id"].apply(set).to_dict()

# family_id -> group_id
fam_to_group = dict(zip(memberships["family_id"], memberships["group_id"]))


def get_top_ri_for_group(group_id, n=3):
    fam_ids = group_families.get(group_id, set())
    sub = ri_trips[ri_trips["bfamily_id"].isin(fam_ids)].copy()
    if sub.empty:
        return "(no R->I triplet data)"
    agg = (
        sub.groupby(["risk_cid", "interv_cid"])["n_triplet_paths"]
        .sum()
        .reset_index()
        .sort_values("n_triplet_paths", ascending=False)
        .head(n)
    )
    lines = []
    for _, r in agg.iterrows():
        rn = risk_names.get(str(int(r["risk_cid"])), f"R{int(r['risk_cid'])}")
        inv = interv_names.get(str(int(r["interv_cid"])), f"I{int(r['interv_cid'])}")
        lines.append(f"  [{rn}] -> [{inv}]  ({int(r['n_triplet_paths'])} paths)")
    return "\n".join(lines)


NAMING_PROMPT = """\
You are naming a causal mechanism group in an AI safety knowledge graph.

CONTEXT:
This group contains {n_frozensets} path signature families ({n_paths} total paths).
Paths in this group share a common set of intermediate body concept clusters.

Core body concept clusters (centroid — most consistently present):
{centroid_decoded}

Most representative frozensets (closest to group centroid):
{closest3_decoded}

Borderline frozensets (farthest from centroid — where the group edges are):
{farthest3_decoded}

Top risk -> intervention pairs this group bridges:
{ri_context}

TASK:
Name the shared causal mechanism that explains WHY these interventions mitigate
these risks via this combination of body concepts.

Your name must complete this sentence naturally and truthfully:
  "The reason why [the intervention] mitigates [the risk] is [YOUR NAME]"

Note: the borderline frozensets show where the group boundary is — your name
should capture what is shared by the closest frozensets while acknowledging
the borderline cases.

RULES:
1. Do NOT start with "via", "through", "by means of", or any preposition
2. Do NOT restate the intervention or risk name
3. DO describe the intermediate mechanism
4. 4-10 words, a noun phrase describing the mechanism
5. Write the test sentence using real risk/intervention names from context above
6. Confirm the sentence is coherent

Respond as valid JSON only:
{{
  "name": "4-10 word causal mechanism noun phrase",
  "description": "one sentence explaining the mechanism",
  "test_sentence": "The reason why [specific intervention] mitigates [specific risk] is [your name]",
  "test_sentence_ok": true/false,
  "test_sentence_reasoning": "brief explanation",
  "borderline_note": "brief note on whether borderline frozensets are consistent with the name",
  "confidence": "high/medium/low"
}}
"""

JUDGE_PROMPT = """\
You are reviewing a proposed causal mechanism group name in an AI safety knowledge graph.

Proposed name: "{name}"
Test sentence: "{test_sentence}"
Author confidence: {confidence}
Borderline note: {borderline_note}

Core body concepts (centroid): {centroid_decoded}
Closest frozensets: {closest3_decoded}
Borderline frozensets: {farthest3_decoded}
Top R->I pairs: {ri_context}

REVIEW:
1. Does the name describe a causal mechanism (not a preposition phrase)?
2. Is the name consistent with BOTH closest AND borderline frozensets? (The borderline ones show group heterogeneity)
3. If borderline frozensets suggest a broader or narrower scope, does the name reflect that?
4. Does the name start with "via", "through", or "by means of"?

Respond as valid JSON only:
{{
  "starts_via_through": true/false,
  "name_fits_closest": true/false,
  "name_fits_borderline": true/false,
  "borderline_consistency_note": "brief note",
  "suggested_revision": "revised name or null",
  "judge_confidence": "high/medium/low"
}}
"""


def call_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=450,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  LLM attempt {attempt + 1} failed: {e}")
            time.sleep(2**attempt)
    return {}


# ── Main naming loop ──────────────────────────────────────────────────────────

results = []

for _, row in groups.sort_values("n_paths_total", ascending=False).iterrows():
    gid = int(row["group_id"])
    n_frozensets = int(row["n_frozensets"])
    n_paths = int(row["n_paths_total"])
    centroid_decoded = str(row["centroid_decoded"])
    closest3_decoded = str(row["closest3_decoded"])
    farthest3_decoded = str(row["farthest3_decoded"])
    ri_context = get_top_ri_for_group(gid)

    print(f"\n-- G{gid:2d} | {n_frozensets} frozensets | {n_paths} paths")
    print(f"   Centroid: {centroid_decoded[:80]}")

    p1 = call_llm(
        NAMING_PROMPT.format(
            n_frozensets=n_frozensets,
            n_paths=n_paths,
            centroid_decoded=centroid_decoded[:600],
            closest3_decoded=closest3_decoded[:800],
            farthest3_decoded=farthest3_decoded[:600],
            ri_context=ri_context,
        )
    )

    name = p1.get("name", "")
    description = p1.get("description", "")
    test_sentence = p1.get("test_sentence", "")
    test_ok = p1.get("test_sentence_ok", True)
    test_reasoning = p1.get("test_sentence_reasoning", "")
    borderline_note = p1.get("borderline_note", "")
    p1_confidence = p1.get("confidence", "")

    print(f"   P1 name: {name}")
    print(f"   Test:    {test_sentence[:100]}")

    p2 = call_llm(
        JUDGE_PROMPT.format(
            name=name,
            test_sentence=test_sentence,
            confidence=p1_confidence,
            borderline_note=borderline_note,
            centroid_decoded=centroid_decoded[:400],
            closest3_decoded=closest3_decoded[:400],
            farthest3_decoded=farthest3_decoded[:400],
            ri_context=ri_context[:300],
        )
    )

    starts_via = p2.get("starts_via_through", False)
    fits_closest = p2.get("name_fits_closest", True)
    fits_borderline = p2.get("name_fits_borderline", True)
    borderline_consistency = p2.get("borderline_consistency_note", "")
    suggested = p2.get("suggested_revision", None)
    judge_conf = p2.get("judge_confidence", "")

    # Use suggested if judge flags issues
    if starts_via or not fits_closest:
        final_name = suggested if suggested else name
        for prefix in ["via ", "through ", "by means of "]:
            if final_name.lower().startswith(prefix):
                final_name = final_name[len(prefix) :].capitalize()
    else:
        final_name = name

    flag = "FLAG" if (starts_via or not fits_closest or not fits_borderline) else "ok"
    print(
        f"   Judge: via={starts_via}, fits_closest={fits_closest}, fits_borderline={fits_borderline}, conf={judge_conf} [{flag}]"
    )
    if suggested:
        print(f"   Suggested: {suggested}")
    print(f"   Final: {final_name}")

    results.append(
        {
            "group_id": gid,
            "n_frozensets": n_frozensets,
            "n_paths_total": n_paths,
            "centroid_decoded": centroid_decoded,
            "llm_name": name,
            "description": description,
            "test_sentence": test_sentence,
            "test_sentence_ok": test_ok,
            "test_sentence_reasoning": test_reasoning,
            "borderline_note": borderline_note,
            "p1_confidence": p1_confidence,
            "judge_starts_via": starts_via,
            "judge_fits_closest": fits_closest,
            "judge_fits_borderline": fits_borderline,
            "borderline_consistency_note": borderline_consistency,
            "suggested_revision": suggested or "",
            "judge_confidence": judge_conf,
            "final_name": final_name,
        }
    )

    time.sleep(0.4)

out_df = pd.DataFrame(results)
out_path = NAMING_DIR / "frozenset_group_names_llm.csv"
out_df.to_csv(out_path, index=False)
print(f"\nWritten: {out_path} ({len(out_df)} rows)")

n_flagged = out_df[
    out_df["judge_starts_via"]
    | ~out_df["judge_fits_closest"]
    | ~out_df["judge_fits_borderline"]
].shape[0]
print(f"Flagged: {n_flagged}/{len(out_df)}")
print("\nFinal names by path count:")
for _, r in out_df.sort_values("n_paths_total", ascending=False).iterrows():
    flag = "F" if (r["judge_starts_via"] or not r["judge_fits_closest"]) else " "
    print(
        f"  [{flag}] G{int(r['group_id']):2d} ({int(r['n_paths_total']):6d} paths): {r['final_name']}"
    )
