"""
Phase 2 Track B2 — LLM Naming Rerun with gpt-5.4-mini + PathbuildB Chain Naming
=================================================================================
Revision plan item 14.

1. Rerun ALL risk + intervention cluster naming with gpt-5.4-mini (was gpt-4o-mini).
2. Name top-40 PathbuildB B-families (consim1) as L2 chain clusters.
   - Prompt: "Intervention [I] addresses risk [R] because [chain mechanism]"
   - Use decoded body component names from optionB_cooccurrence_families_consim1.csv
     and bodysubtype_cluster_representatives.csv

PathbuildA chain naming is NOT rerun here (PathbuildA is rejected per A2 decision).
PathbuildB B-families replace PathbuildA chain clusters as the L2 chain taxonomy.

Outputs (phase2_results/step5_naming/):
  risk_cluster_names_llm_v2.csv
  intervention_cluster_names_llm_v2.csv
  pathbuildB_chain_names_llm.csv
  all_clusters_naming_detail_v2.csv
"""

import csv
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
load_dotenv(BASE / ".env", override=True)
# Normalise lowercase key
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = os.environ["openai_api_key"]

PKL_DIR = BASE / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
PATHS_DIR = BASE / "phase1_rawpathsfiles"
STEP4_DIR = BASE / "phase2_results/step4_finalanalysis"
TABLES_DIR = STEP4_DIR / "step4_cluster_tables"
OUT_DIR = BASE / "phase2_results/step5_naming"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL = "gpt-5.4-mini"
N_TOP_NODES = 10  # number of representative nodes per cluster for naming
N_BFAMILIES = 40  # number of top PathbuildB families to name

client = OpenAI()

# ─── STEP 1: Load PKL ─────────────────────────────────────────────────────────
print("Loading cluster_memberships.pkl ...")
t0 = time.time()
with open(PKL_DIR / "cluster_memberships.pkl", "rb") as f:
    cm = pickle.load(f)
print(f"  {len(cm)} keys  ({time.time() - t0:.1f}s)")

print("Loading node_attrs.pkl ...")
t1 = time.time()
with open(PKL_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
print(f"  {len(node_attrs)} nodes  ({time.time() - t1:.1f}s)")

# ─── STEP 2: Build valid_pathway_nodes from path file ─────────────────────────
print("Building valid_pathway_nodes from unconstrained path file ...")
t2 = time.time()
valid_pathway_nodes = set()
paths_file = PATHS_DIR / "paths_unconstrained_sim0.9.jsonl"
with open(paths_file) as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj.get("path") or []]
        if not path:
            continue
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) >= 3:
            valid_pathway_nodes.update(path)
print(f"  {len(valid_pathway_nodes):,} VPN nodes  ({time.time() - t2:.1f}s)")


# ─── STEP 3: Helpers ──────────────────────────────────────────────────────────
def get_embedding(nid):
    raw = node_attrs.get(nid, {}).get("embedding", "")
    if not raw:
        return None
    try:
        return np.array(
            [float(x) for x in str(raw).strip("<>").split(",")], dtype=np.float32
        )
    except Exception:
        return None


def compute_centroid(node_ids):
    embs = [get_embedding(nid) for nid in node_ids]
    embs = [e for e in embs if e is not None]
    if not embs:
        return None
    arr = np.stack(embs)
    c = arr.mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def top_n_by_centroid_sim(node_ids, centroid, n=10):
    sims = []
    for nid in node_ids:
        e = get_embedding(nid)
        if e is None:
            continue
        e_norm = e / (np.linalg.norm(e) + 1e-8)
        sims.append((float(np.dot(e_norm, centroid)), nid))
    sims.sort(reverse=True)
    return [nid for _, nid in sims[:n]]


def format_nodes_for_prompt(node_ids):
    lines = []
    for i, nid in enumerate(node_ids, 1):
        attrs = node_attrs.get(nid, {})
        name = attrs.get("name", f"[node {nid}]")
        desc = (attrs.get("description") or "")[:150]
        cat = attrs.get("concept_category") or attrs.get("type", "")
        lines.append(f"{i}. [{cat}] {name}: {desc}")
    return "\n".join(lines)


def get_cluster_dict_vpn_filtered(
    node_type, ec=0.9, mode="unconstrained", algo="agglomerative"
):
    """Get {cluster_id: [node_ids]} filtered to VPN."""
    result = {}
    for (e, m, nt, a, cid), members in cm.items():
        try:
            e_float = float(e)
        except (ValueError, TypeError):
            continue
        if e_float == ec and m == mode and nt == node_type and a == algo:
            filtered = [nid for nid in members if nid in valid_pathway_nodes]
            if filtered:
                result[int(cid)] = filtered
    return result


# ─── STEP 4: LLM prompts ──────────────────────────────────────────────────────
TYPE_INSTRUCTIONS = {
    "risk": (
        "What failure mode, danger, or problem does this cluster represent? "
        "Frame as the core AI safety risk these nodes share."
    ),
    "intervention": (
        "What action, approach, or technique does this cluster represent? "
        "Frame as the concrete intervention or mitigation strategy."
    ),
    "chain": (
        "What causal mechanism connects a risk to an intervention through these intermediate concepts? "
        "Frame STRICTLY as: 'These chains connect AI safety risks to interventions by reasoning about [mechanism].' "
        "The mechanism must be specific (e.g., 'reward specification and human preference modeling') NOT a risk re-statement."
    ),
}


def naming_prompt(cluster_type, nodes_text):
    return (
        f"You are naming clusters in an AI safety knowledge graph.\n\n"
        f"Cluster type: {cluster_type}\n"
        f"Representative nodes (ordered by semantic centrality, most representative first):\n"
        f"{nodes_text}\n\n"
        f"Task: Generate a concise cluster name and description.\n"
        f"- Name: 5-10 words, captures the shared theme across ALL nodes listed\n"
        f"- Description: 1 sentence (max 30 words), captures the causal logic or shared concept\n"
        f"- {TYPE_INSTRUCTIONS[cluster_type]}\n\n"
        f'Respond as valid JSON only: {{"name": "...", "description": "..."}}'
    )


def judge_prompt(cluster_type, nodes_text, name, desc):
    return (
        f"You are reviewing a cluster name for an AI safety knowledge graph.\n\n"
        f"Cluster type: {cluster_type}\n"
        f'Proposed name: "{name}"\n'
        f'Proposed description: "{desc}"\n\n'
        f"Representative nodes:\n{nodes_text}\n\n"
        f"Review criteria:\n"
        f"1. Does the name accurately capture the shared theme? (accurate)\n"
        f"2. Do the nodes span more than 2 distinct themes? (split_candidate)\n"
        f"3. Is the name specific enough to distinguish from adjacent clusters? (confidence)\n"
        f"4. Suggest improved name if needed (suggested_revision).\n\n"
        f"Respond as valid JSON only:\n"
        f'{{"accurate": true/false, "issues": "brief or null", '
        f'"split_candidate": true/false, "confidence": "high/medium/low", '
        f'"suggested_revision": "revised or null"}}'
    )


def bfamily_naming_prompt(family_id, n_paths, decoded_components_text):
    return (
        f"You are naming a PathbuildB chain family in an AI safety knowledge graph.\n\n"
        f"A PathbuildB family is a group of {n_paths} argument chains (paths from risk → intervention) "
        f"that share the same combination of intermediate body concept clusters.\n\n"
        f"Decoded chain body components (intermediate reasoning concepts from most to least common subtype):\n"
        f"{decoded_components_text}\n\n"
        f"Task: What causal MECHANISM do these intermediate concepts describe?\n"
        f"- Name: 5-10 words capturing the mechanistic reasoning theme\n"
        f"- Description: 1 sentence starting with 'These chains connect AI safety risks to interventions by reasoning about ...'\n"
        f"- CRITICAL: The name and description MUST describe the CAUSAL MECHANISM, NOT re-state a risk.\n"
        f"  BAD: 'Catastrophic AI misalignment risk' (re-states a risk)\n"
        f"  GOOD: 'Reward specification and preference learning' (describes the mechanism)\n"
        f"  BAD: 'Existential risk from advanced AI' (re-states a risk)\n"
        f"  GOOD: 'Adversarial robustness training for neural networks' (describes the mechanism)\n\n"
        f'Respond as valid JSON only: {{"name": "...", "description": "..."}}'
    )


def call_llm(prompt, max_tokens=300, retries=3):
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


# ─── STEP 5: Process risk + intervention clusters ─────────────────────────────
FIELDS = [
    "cluster_type",
    "cluster_id",
    "n_members",
    "llm_name",
    "llm_description",
    "judge_accurate",
    "judge_issues",
    "judge_split_candidate",
    "judge_confidence",
    "suggested_revision",
    "final_name",
    "top5_node_names",
]


def process_clusters(cluster_type, cluster_dict):
    rows = []
    total = len(cluster_dict)
    for i, cid in enumerate(sorted(cluster_dict.keys()), 1):
        members = cluster_dict[cid]
        centroid = compute_centroid(members)
        if centroid is None:
            print(f"  WARNING: {cluster_type} cluster {cid}: no embeddings, skipping")
            continue
        top_ids = top_n_by_centroid_sim(members, centroid, n=N_TOP_NODES)
        nodes_text = format_nodes_for_prompt(top_ids)
        top5_names = " | ".join(
            node_attrs.get(nid, {}).get("name", "") for nid in top_ids[:5]
        )

        # Pass 1: name
        p1 = call_llm(naming_prompt(cluster_type, nodes_text))
        llm_name = p1.get("name", "")
        llm_desc = p1.get("description", "")

        # Pass 2: judge
        p2 = call_llm(judge_prompt(cluster_type, nodes_text, llm_name, llm_desc))
        judge_acc = p2.get("accurate", True)
        judge_issues = p2.get("issues", "")
        judge_split = p2.get("split_candidate", False)
        judge_conf = p2.get("confidence", "medium")
        suggested = p2.get("suggested_revision", "")

        final = suggested if suggested and not judge_acc else llm_name
        rows.append(
            {
                "cluster_type": cluster_type,
                "cluster_id": cid,
                "n_members": len(members),
                "llm_name": llm_name,
                "llm_description": llm_desc,
                "judge_accurate": judge_acc,
                "judge_issues": judge_issues or "",
                "judge_split_candidate": judge_split,
                "judge_confidence": judge_conf,
                "suggested_revision": suggested or "",
                "final_name": final,
                "top5_node_names": top5_names,
            }
        )
        status = "✓" if judge_conf == "high" else "△"
        print(f"  [{i}/{total}] {cluster_type} C{cid}: {status} '{final[:60]}'")
    return rows


print("\n=== Risk cluster naming ===")
risk_dict = get_cluster_dict_vpn_filtered("risk")
print(f"  {len(risk_dict)} risk clusters")
risk_rows = process_clusters("risk", risk_dict)

print("\n=== Intervention cluster naming ===")
interv_dict = get_cluster_dict_vpn_filtered("intervention")
print(f"  {len(interv_dict)} intervention clusters")
interv_rows = process_clusters("intervention", interv_dict)

# ─── STEP 6: PathbuildB B-family chain naming ─────────────────────────────────
print(f"\n=== PathbuildB B-family chain naming (top-{N_BFAMILIES}) ===")

# Load decoded B-family data
ob_families = pd.read_csv(TABLES_DIR / "optionB_cooccurrence_families_consim1.csv")
ob_top20_decoded = pd.read_csv(TABLES_DIR / "optionB_top20_decoded_consim1.csv")
bcr = pd.read_csv(TABLES_DIR / "bodysubtype_cluster_representatives.csv")
prefix_to_name = dict(zip(bcr["prefix_key"], bcr["rep_name"].str[:80]))

# Get top-40 families
top_families = ob_families.nlargest(N_BFAMILIES, "n_paths").reset_index(drop=True)

# Build decoded_top20 lookup by rank
decoded_by_rank = {}
for _, row in ob_top20_decoded.iterrows():
    decoded_by_rank[int(row["rank"]) - 1] = {
        "n_paths": row["n_paths"],
        "signature_str": row["signature_str"],
        "decoded": row["decoded_chain_components"],
    }


def decode_signature_str(sig_str):
    """Decode 'de:15 & im:4 & pr:6 ...' → list of 'prefix_key: rep_name' lines."""
    parts = [p.strip() for p in sig_str.split("&")]
    lines = []
    for part in parts:
        rep = prefix_to_name.get(part, f"[{part}]")
        lines.append(f"  {part}: {rep}")
    return "\n".join(lines)


chain_rows = []
for rank0 in range(N_BFAMILIES):
    row = top_families.iloc[rank0]
    n_paths = int(row["n_paths"])
    sig_str = str(row["signature_str"])

    # Use pre-decoded text if available (top-20), else decode from scratch
    if rank0 in decoded_by_rank:
        decoded_text = str(decoded_by_rank[rank0].get("decoded", ""))
        # Also add decoded version
        decoded_fallback = decode_signature_str(sig_str)
        decoded_components_text = (
            decoded_text + "\n\nAlternative decoded:\n" + decoded_fallback
        )
    else:
        decoded_components_text = decode_signature_str(sig_str)

    prompt = bfamily_naming_prompt(rank0 + 1, n_paths, decoded_components_text[:1500])
    p1 = call_llm(prompt, max_tokens=200)
    llm_name = p1.get("name", "")
    llm_desc = p1.get("description", "")

    # Judge check (simpler for B-families — just flag if it re-states a risk)
    judge_prompt_text = (
        f'Chain family name: "{llm_name}"\n'
        f'Description: "{llm_desc}"\n\n'
        f"Does this name describe a CAUSAL MECHANISM or does it re-state an AI safety RISK? "
        f"If it re-states a risk (contains 'catastrophic', 'existential', 'misalignment' as the main theme), "
        f"flag as inaccurate and provide a better mechanism-focused name.\n"
        f"Chain components:\n{decoded_components_text[:800]}\n\n"
        f'Respond as valid JSON: {{"accurate": true/false, "issues": "...", '
        f'"confidence": "high/medium/low", "suggested_revision": "... or null"}}'
    )
    p2 = call_llm(judge_prompt_text, max_tokens=200)
    judge_acc = p2.get("accurate", True)
    judge_issues = p2.get("issues", "")
    judge_conf = p2.get("confidence", "medium")
    suggested = p2.get("suggested_revision", "")

    final = suggested if suggested and not judge_acc else llm_name

    chain_rows.append(
        {
            "cluster_type": "chain_pathbuildB",
            "cluster_id": rank0,
            "n_members": n_paths,
            "llm_name": llm_name,
            "llm_description": llm_desc,
            "judge_accurate": judge_acc,
            "judge_issues": judge_issues or "",
            "judge_split_candidate": False,
            "judge_confidence": judge_conf,
            "suggested_revision": suggested or "",
            "final_name": final,
            "top5_node_names": sig_str[:200],  # Store signature as proxy for top5
        }
    )
    status = "✓" if judge_conf == "high" else "△"
    print(
        f"  [rank {rank0 + 1}/{N_BFAMILIES}] {status} '{final[:70]}' ({n_paths} paths)"
    )


# ─── STEP 7: Write outputs ─────────────────────────────────────────────────────
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}  ({len(rows)} rows)")


print("\nWriting outputs ...")
write_csv(OUT_DIR / "risk_cluster_names_llm_v2.csv", risk_rows)
write_csv(OUT_DIR / "intervention_cluster_names_llm_v2.csv", interv_rows)
write_csv(OUT_DIR / "pathbuildB_chain_names_llm.csv", chain_rows)
write_csv(
    OUT_DIR / "all_clusters_naming_detail_v2.csv", risk_rows + interv_rows + chain_rows
)

# ─── STEP 8: Summary ──────────────────────────────────────────────────────────
print("\n=== B2 Naming Summary ===")
for cluster_type, rows in [
    ("risk", risk_rows),
    ("intervention", interv_rows),
    ("chain_pathbuildB", chain_rows),
]:
    n_high = sum(1 for r in rows if r["judge_confidence"] == "high")
    n_split = sum(1 for r in rows if r["judge_split_candidate"])
    n_inaccurate = sum(1 for r in rows if not r["judge_accurate"])
    print(
        f"  {cluster_type}: {len(rows)} named, "
        f"{n_high} high-conf, {n_split} split-candidates, {n_inaccurate} inaccurate"
    )

print("\nTop-10 PathbuildB chain names:")
for i, row in enumerate(chain_rows[:10]):
    print(f"  B-Fam {i + 1} ({row['n_members']} paths): {row['final_name']}")

print("\nDone.")
