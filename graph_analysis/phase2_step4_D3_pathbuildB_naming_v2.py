"""
Phase 2 Step 4 D3 — PathbuildB chain naming rerun with "via [mechanism]" framing.

Redesigns the naming prompt so that chain family names fit the template:
  "[Risk type] addressed via [intermediate mechanism]"
  OR describe the reasoning chain as "through [concept1 → concept2 → ...]"

Reruns gpt-5.4-mini 2-pass naming on all top-40 B-families.

Output: step5_naming/pathbuildB_chain_names_llm_v2.csv
"""

import json
import os
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ─── Paths + API ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
load_dotenv(BASE / ".env", override=True)
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = os.environ["openai_api_key"]

TABLES_DIR = BASE / "phase2_results/step4_finalanalysis/step4_cluster_tables"
NAMING_DIR = BASE / "phase2_results/step5_naming"

MODEL = "gpt-5.4-mini"
N_FAMILIES = 40

client = OpenAI()

# ─── Load data ─────────────────────────────────────────────────────────────────
fam_df = pd.read_csv(TABLES_DIR / "optionB_cooccurrence_families_consim1.csv")
top40 = fam_df.nlargest(N_FAMILIES, "n_paths").reset_index(drop=True)

# Load decoded components for top-20
try:
    decoded_df = pd.read_csv(TABLES_DIR / "optionB_top20_decoded_consim1.csv")
    decoded_by_rank = {
        int(r["rank"]) - 1: str(r["decoded_chain_components"])
        for _, r in decoded_df.iterrows()
    }
except Exception:
    decoded_df = pd.DataFrame()
    decoded_by_rank = {}

# Load existing v1 names for reference
try:
    v1_df = pd.read_csv(NAMING_DIR / "pathbuildB_chain_names_llm.csv")
    v1_names = {
        int(r["cluster_id"]): str(r.get("final_name") or r.get("llm_name", ""))
        for _, r in v1_df.iterrows()
    }
except Exception:
    v1_names = {}

# Load component representative names
try:
    bcr = pd.read_csv(TABLES_DIR / "bodysubtype_cluster_representatives.csv")
    prefix_to_name = dict(zip(bcr["prefix_key"], bcr["rep_name"].str[:80]))
except Exception:
    prefix_to_name = {}


# ─── Helper: decode signature ─────────────────────────────────────────────────
def decode_signature(sig_str):
    parts = [p.strip() for p in sig_str.split("&")]
    lines = []
    for part in parts:
        rep = prefix_to_name.get(part, f"[{part}]")
        lines.append(f"  {part}: {rep}")
    return "\n".join(lines)


# ─── LLM helpers ──────────────────────────────────────────────────────────────
def call_llm(prompt, max_tokens=250, retries=3):
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


VIA_NAMING_PROMPT = """\
You are naming a PathbuildB chain family in an AI safety knowledge graph.

A PathbuildB chain family groups {n_paths} argument chains that all share the same \
combination of intermediate body concept clusters. The chain structure is:
  RISK CLUSTER → [intermediate body concepts] → INTERVENTION CLUSTER

Existing v1 name (for reference only, do NOT copy): "{v1_name}"

Decoded body components (the intermediate reasoning steps, each is a concept cluster):
{decoded_text}

Task: Give this chain family a name that makes the **intermediate mechanism** visible.

Required format: The name must describe WHAT THE CHAIN PASSES THROUGH — the mechanism \
or reasoning bridge between risk and intervention.

Good examples (notice: the name describes the mechanism, not the endpoint):
  "via reward specification and adversarial robustness testing"
  "through interpretability tools and model transparency audits"
  "via preference learning and RLHF feedback loops"
  "through compute governance and hardware chokepoints"
  "via scalable oversight and debate mechanisms"

Bad examples (do not copy):
  "Existential risk from misaligned AI" (re-states the risk)
  "Expand AI safety research funding" (re-states the intervention)
  "Funding and training to build AI safety capacity" (v1 name — too endpoint-focused)

Format:
  - Name: 5–10 words starting with "via" or "through" OR a short phrase like \
"[Mechanism] chain" — the mechanism must be the subject, not the risk or intervention
  - Description: one sentence explaining what intermediate concepts these chains reason about

Respond as valid JSON only: {{"name": "...", "description": "..."}}
"""

VIA_JUDGE_PROMPT = """\
You are reviewing a PathbuildB chain family name in an AI safety knowledge graph.

Proposed name: "{name}"
Description: "{desc}"

Chain body components (intermediate concepts):
{decoded_text}

Review:
1. Does the name describe the INTERMEDIATE MECHANISM (not just a risk or intervention name)?
2. Is "via" or "through" or a mechanism keyword prominent in the name?
3. Would this name help distinguish this chain pathway from adjacent ones?

If the name re-states a RISK (catastrophic, existential, misalignment as main theme) or \
re-states an INTERVENTION (fund, expand, deploy X as main theme) rather than describing \
the intermediate mechanism, flag as inaccurate.

Respond as valid JSON only:
{{"accurate": true/false, "issues": "brief or null", \
"confidence": "high/medium/low", "suggested_revision": "revised name or null"}}
"""

# ─── Main naming loop ─────────────────────────────────────────────────────────
print(f"Running D3 PathbuildB naming v2 on top-{N_FAMILIES} families ...")
rows = []

for rank0, row in top40.iterrows():
    n_paths = int(row["n_paths"])
    sig_str = str(row["signature_str"])
    v1_name = v1_names.get(rank0, "")

    # Build decoded text
    if rank0 in decoded_by_rank:
        decoded_text = decoded_by_rank[rank0][:1200]
    else:
        decoded_text = decode_signature(sig_str)

    # Pass 1: name
    prompt1 = VIA_NAMING_PROMPT.format(
        n_paths=n_paths,
        v1_name=v1_name,
        decoded_text=decoded_text,
    )
    p1 = call_llm(prompt1, max_tokens=200)
    llm_name = p1.get("name", "")
    llm_desc = p1.get("description", "")

    # Pass 2: judge
    prompt2 = VIA_JUDGE_PROMPT.format(
        name=llm_name,
        desc=llm_desc,
        decoded_text=decoded_text[:800],
    )
    p2 = call_llm(prompt2, max_tokens=200)
    judge_acc = p2.get("accurate", True)
    judge_issues = p2.get("issues", "")
    judge_conf = p2.get("confidence", "medium")
    suggested = p2.get("suggested_revision", "")

    final = suggested if suggested and not judge_acc else llm_name

    rows.append(
        {
            "cluster_type": "chain_pathbuildB",
            "cluster_id": rank0,
            "n_members": n_paths,
            "signature_str": sig_str,
            "v1_name": v1_name,
            "llm_name": llm_name,
            "llm_description": llm_desc,
            "judge_accurate": judge_acc,
            "judge_issues": judge_issues or "",
            "judge_confidence": judge_conf,
            "suggested_revision": suggested or "",
            "final_name": final,
        }
    )

    status = "✓" if judge_conf == "high" else ("△" if judge_conf == "medium" else "✗")
    print(f"  [{rank0 + 1:2d}/{N_FAMILIES}] {status} '{final[:70]}'")

# ─── Save ──────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(rows)
out_path = NAMING_DIR / "pathbuildB_chain_names_llm_v2.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved {out_path} ({len(out_df)} rows)")

# Quick summary
high = (out_df["judge_confidence"] == "high").sum()
accurate = out_df["judge_accurate"].sum()
print(
    f"High confidence: {high}/{N_FAMILIES}  |  Judge accurate: {accurate}/{N_FAMILIES}"
)
print("\nTop-10 v2 names:")
for _, r in out_df.head(10).iterrows():
    print(f"  B{int(r['cluster_id'])}: {r['final_name']}")
