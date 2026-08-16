"""
Phase 2 Step 4 D4 — PathbuildB chain naming v3 with causal chain framing.

Item 7 from step45_revision4_inputs.txt:

The "via [mechanism]" framing is wrong — e.g. "MF20 is via RLHF alignment inadequacy"
implies the intervention causes RLHF alignment inadequacy (something bad), which
mitigates the risk. That is backwards.

New framing: the MF body name should complete the sentence:
  "The reason why [intervention cluster name] mitigates [risk cluster name] is [MF title]"

Requirements:
  - Name must be 4–9 words describing the causal mechanism
  - No "via" or "through" in the name
  - The LLM must construct and confirm the test sentence makes sense

Output: step5_naming/pathbuildB_chain_names_llm_v3.csv
"""

import json
import os
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

BASE = Path(__file__).parent
load_dotenv(BASE / ".env", override=True)
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = os.environ["openai_api_key"]

TABLES_DIR = BASE / "phase2_results/step4_finalanalysis/step4_cluster_tables"
NAMING_DIR = BASE / "phase2_results/step5_naming"
CONN_DIR = BASE / "phase2_results/step4_finalanalysis/step4_connectivity"

MODEL = "gpt-4.1-mini"
N_FAMILIES = 40

client = OpenAI()

# ─── Load data ──────────────────────────────────────────────────────────────
fam_df = pd.read_csv(TABLES_DIR / "optionB_cooccurrence_families_consim1.csv")
top40 = fam_df.nlargest(N_FAMILIES, "n_paths").reset_index(drop=True)

# Load v2 names for reference (so LLM can see what was tried before)
try:
    v2_df = pd.read_csv(NAMING_DIR / "pathbuildB_chain_names_llm_v2.csv")
    v2_names = {
        int(r["cluster_id"]): str(r.get("final_name") or r.get("llm_name", ""))
        for _, r in v2_df.iterrows()
    }
except Exception:
    v2_names = {}

# Load decoded components for top-40
try:
    decoded_df = pd.read_csv(TABLES_DIR / "optionB_top20_decoded_consim1.csv")
    decoded_by_rank = {
        int(r["rank"]) - 1: str(r["decoded_chain_components"])
        for _, r in decoded_df.iterrows()
    }
except Exception:
    decoded_by_rank = {}

# Load component representative names
try:
    bcr = pd.read_csv(TABLES_DIR / "bodysubtype_cluster_representatives.csv")
    prefix_to_name = dict(zip(bcr["prefix_key"], bcr["rep_name"]))
except Exception:
    prefix_to_name = {}

# Load risk-to-intervention connectivity for context (what top R→I pairs does this MF connect?)
try:
    ri_trip_path = CONN_DIR / "ri_triplets_consim1.csv"
    if not ri_trip_path.exists():
        ri_trip_path = CONN_DIR / "ri_meta_triplets_consim1.csv"
    ri_trips = pd.read_csv(ri_trip_path) if ri_trip_path.exists() else pd.DataFrame()
except Exception:
    ri_trips = pd.DataFrame()

# Load risk/intervention final names
risk_df = pd.read_csv(NAMING_DIR / "risk_cluster_names_llm_v2.csv")
interv_df = pd.read_csv(NAMING_DIR / "intervention_cluster_names_llm_v2.csv")
risk_name_col = "final_name" if "final_name" in risk_df.columns else "llm_name"
interv_name_col = "final_name" if "final_name" in interv_df.columns else "llm_name"
risk_names = dict(zip(risk_df["cluster_id"].astype(str), risk_df[risk_name_col]))
interv_names = dict(
    zip(interv_df["cluster_id"].astype(str), interv_df[interv_name_col])
)


def decode_signature(sig_str):
    parts = [p.strip() for p in sig_str.split("&")]
    lines = []
    for part in parts:
        rep = prefix_to_name.get(part, f"[{part}]")
        lines.append(f"  {part}: {rep}")
    return "\n".join(lines)


# Get top risk/interv pairs for a given family (by path count)
def get_top_ri_context(family_id, n=3):
    if ri_trips.empty:
        return ""
    col_options = ["b_family_id", "family_id", "bfamily_id"]
    fam_col = next((c for c in col_options if c in ri_trips.columns), None)
    if fam_col is None:
        return ""
    sub = ri_trips[ri_trips[fam_col] == family_id]
    if sub.empty:
        return ""
    path_col = next((c for c in ["n_paths", "n_paths_c1"] if c in sub.columns), None)
    if path_col:
        sub = sub.sort_values(path_col, ascending=False)
    lines = []
    for _, r in sub.head(n).iterrows():
        rid = str(int(r.get("risk_cid", r.get("risk_cluster", -1))))
        iid = str(int(r.get("interv_cid", r.get("interv_cluster", -1))))
        rname = risk_names.get(rid, f"R{rid}")
        iname = interv_names.get(iid, f"I{iid}")
        lines.append(f"  R{rid} [{rname}] -> I{iid} [{iname}]")
    return "\n".join(lines)


# ─── LLM helpers ─────────────────────────────────────────────────────────────
def call_llm(prompt, max_tokens=350, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  WARNING: LLM call failed attempt {attempt + 1}: {e}")
            time.sleep(2**attempt)
    return {}


CAUSAL_NAMING_PROMPT = """\
You are naming a PathbuildB chain mechanism family in an AI safety knowledge graph.

CONTEXT:
A chain family groups {n_paths} reasoning paths that all pass through the same \
combination of intermediate body concept clusters, connecting risk clusters to \
intervention clusters.

Chain structure:  RISK CLUSTER -> [intermediate mechanism body] -> INTERVENTION CLUSTER

Top risk-to-intervention pairs this chain family connects:
{ri_context}

Intermediate body concept clusters (the mechanism being named):
{decoded_text}

Previous v2 name (WRONG — uses "via"/"through" framing): "{v2_name}"

TASK:
Name the intermediate mechanism that causally explains why interventions mitigate risks \
via this chain.

REQUIREMENT: The final name must complete this sentence naturally and truthfully:
  "The reason why [the intervention] mitigates [the risk] is [YOUR NAME]"

Example: If the intervention is "RLHF fine-tuning" and the risk is "deceptive alignment", \
a good name would be "reward signal alignment with human preferences" because:
  "The reason why RLHF fine-tuning mitigates deceptive alignment is \
reward signal alignment with human preferences" — this makes sense.

RULES:
1. Do NOT use "via", "through", "by means of" as the first word
2. Do NOT restate the intervention name or risk name
3. DO describe the intermediate mechanism — what it IS, not where it comes from
4. Name should be 4–9 words, a noun phrase describing the mechanism
5. You MUST write out the test sentence and confirm it makes sense

Respond as valid JSON only:
{{
  "name": "4-9 word mechanism noun phrase",
  "description": "one sentence explaining the mechanism",
  "test_sentence": "The reason why [specific intervention] mitigates [specific risk] is [your name]",
  "test_sentence_makes_sense": true/false,
  "test_sentence_reasoning": "brief explanation of why the sentence is coherent or not"
}}
"""

CAUSAL_JUDGE_PROMPT = """\
You are reviewing a PathbuildB chain mechanism name in an AI safety knowledge graph.

Proposed name: "{name}"
Test sentence: "{test_sentence}"
Test sentence makes sense (author's assessment): {test_ok}

Intermediate body concepts:
{decoded_text}

REVIEW:
1. Does the name describe an INTERMEDIATE MECHANISM (not an intervention or risk endpoint)?
2. Does the name start with "via", "through", or "by means of"? If so, flag as non-compliant.
3. Does the test sentence read as a plausible causal explanation?
4. Would this name help distinguish this chain family from adjacent ones?

If the name uses "via"/"through" as first word, or re-states a risk/intervention \
rather than the mechanism, flag as inaccurate and suggest a revision.

Respond as valid JSON only:
{{
  "accurate": true/false,
  "starts_with_via_through": true/false,
  "issues": "brief description or null",
  "confidence": "high/medium/low",
  "suggested_revision": "revised name without via/through, or null"
}}
"""

# ─── Main naming loop ─────────────────────────────────────────────────────────
print(
    f"Running D4 PathbuildB naming v3 (causal chain framing) on top-{N_FAMILIES} families ..."
)
rows = []

for rank0, row in top40.iterrows():
    n_paths = int(row["n_paths"])
    sig_str = str(row["signature_str"])
    v2_name = v2_names.get(rank0, "")

    # Build decoded text
    if rank0 in decoded_by_rank:
        decoded_text = decoded_by_rank[rank0][:1200]
    else:
        decoded_text = decode_signature(sig_str)

    # Get R→I context
    ri_context = get_top_ri_context(rank0)
    if not ri_context:
        ri_context = "(no R-I triplet data available)"

    # Pass 1: name with causal framing
    prompt1 = CAUSAL_NAMING_PROMPT.format(
        n_paths=n_paths,
        ri_context=ri_context,
        decoded_text=decoded_text,
        v2_name=v2_name,
    )
    p1 = call_llm(prompt1, max_tokens=350)
    llm_name = p1.get("name", "")
    llm_desc = p1.get("description", "")
    test_sentence = p1.get("test_sentence", "")
    test_ok = p1.get("test_sentence_makes_sense", True)
    test_reasoning = p1.get("test_sentence_reasoning", "")

    # Pass 2: judge
    prompt2 = CAUSAL_JUDGE_PROMPT.format(
        name=llm_name,
        test_sentence=test_sentence,
        test_ok=test_ok,
        decoded_text=decoded_text[:800],
    )
    p2 = call_llm(prompt2, max_tokens=250)
    judge_acc = p2.get("accurate", True)
    starts_via = p2.get("starts_with_via_through", False)
    judge_issues = p2.get("issues", "")
    judge_conf = p2.get("confidence", "medium")
    suggested = p2.get("suggested_revision", "")

    # If judge flags "via"/"through" in name, use suggested revision
    if starts_via or not judge_acc:
        final = suggested if suggested else llm_name
        # If final still starts with "via"/"through", strip it
        for prefix in ["via ", "through ", "by means of "]:
            if final.lower().startswith(prefix):
                final = final[len(prefix) :].capitalize()
    else:
        final = llm_name

    rows.append(
        {
            "cluster_type": "chain_pathbuildB",
            "cluster_id": rank0,
            "n_members": n_paths,
            "signature_str": sig_str,
            "v2_name": v2_name,
            "llm_name": llm_name,
            "llm_description": llm_desc,
            "test_sentence": test_sentence,
            "test_sentence_ok": test_ok,
            "test_sentence_reasoning": test_reasoning,
            "judge_accurate": judge_acc,
            "judge_starts_via_through": starts_via,
            "judge_issues": judge_issues or "",
            "judge_confidence": judge_conf,
            "suggested_revision": suggested or "",
            "final_name": final,
        }
    )

    status = (
        "✓"
        if judge_conf == "high" and judge_acc
        else ("△" if judge_conf == "medium" else "✗")
    )
    via_flag = " [VIA!]" if starts_via else ""
    print(f"  [{rank0 + 1:2d}/{N_FAMILIES}] {status}{via_flag} '{final[:70]}'")

# ─── Save ────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(rows)
out_path = NAMING_DIR / "pathbuildB_chain_names_llm_v3.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved {out_path} ({len(out_df)} rows)")

high = (out_df["judge_confidence"] == "high").sum()
accurate = out_df["judge_accurate"].sum()
via_flagged = out_df["judge_starts_via_through"].sum()
print(
    f"High confidence: {high}/{N_FAMILIES}  |  Judge accurate: {accurate}/{N_FAMILIES}  |  Via-flagged: {via_flagged}"
)
print("\nTop-10 v3 names:")
for _, r in out_df.head(10).iterrows():
    print(f"  B{int(r['cluster_id'])}: {r['final_name']}")
    print(f"    Test: {r['test_sentence'][:100]}")
