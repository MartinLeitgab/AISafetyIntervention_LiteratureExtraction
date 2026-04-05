"""
Phase 2 Step 5 Subcluster Naming
For each of the 36 triggered clusters in subcluster_summary.csv:
- Re-run agglomerative k=5 subclustering on valid_pathway_nodes-filtered members
- Get top-5 representative nodes per subcluster
- Run 2-pass LLM naming (gpt-4o-mini)
- Save outputs
"""

import csv
import json
import os
import pickle
import sys
import time

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/mnt/c/Users/malei/0_project_work/eleutherAI_SOAR_step1knowledgegraphcreation/AISafetyIntervention_LiteratureExtraction"

SUBCLUSTER_SUMMARY_CSV = os.path.join(
    PROJECT_ROOT,
    "graph_analysis/phase2_results/step4_finalanalysis/step4_subclusters/subcluster_summary.csv",
)
CLUSTER_MEMBERSHIPS_PKL = os.path.join(
    PROJECT_ROOT,
    "graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/cluster_memberships.pkl",
)
NODE_ATTRS_PKL = os.path.join(
    PROJECT_ROOT,
    "graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_node_attributes.pkl",
)
PATHS_JSONL = os.path.join(
    PROJECT_ROOT,
    "graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl",
)
RISK_NAMES_CSV = os.path.join(
    PROJECT_ROOT,
    "graph_analysis/phase2_results/step5_naming/risk_cluster_names_llm.csv",
)
INTERVENTION_NAMES_CSV = os.path.join(
    PROJECT_ROOT,
    "graph_analysis/phase2_results/step5_naming/intervention_cluster_names_llm.csv",
)

OUTPUT_DIR_STEP4 = os.path.join(
    PROJECT_ROOT,
    "graph_analysis/phase2_results/step4_finalanalysis/step4_subclusters",
)
OUTPUT_DIR_STEP5 = os.path.join(
    PROJECT_ROOT, "graph_analysis/phase2_results/step5_naming"
)

OUTPUT_CSV_STEP4 = os.path.join(OUTPUT_DIR_STEP4, "subcluster_names_llm.csv")
OUTPUT_CSV_STEP5 = os.path.join(OUTPUT_DIR_STEP5, "subcluster_naming_detail.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_embedding(emb_str):
    if isinstance(emb_str, np.ndarray):
        return emb_str.astype(np.float32)
    s = str(emb_str).strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    v = np.array(vals, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_parent_names():
    """Return dict: (cluster_type, cluster_id_str) -> final_name"""
    names = {}
    for csv_path, ctype in [
        (RISK_NAMES_CSV, "risk"),
        (INTERVENTION_NAMES_CSV, "intervention"),
    ]:
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found")
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = str(row.get("cluster_id", "")).strip()
                fname = row.get("final_name", row.get("llm_name", "")).strip()
                names[(ctype, cid)] = fname
    return names


def call_llm(client, messages, max_tokens=200, retries=3):
    """Call gpt-4o-mini and return parsed JSON dict, or None on failure."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            content = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    ln for ln in lines if not ln.startswith("```")
                ).strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(
                f"  JSON parse error (attempt {attempt + 1}): {e} — raw: {content[:200]}"
            )
            if attempt < retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"  LLM call error (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def pass1_prompt(parent_name, node_type, sub_id, n_nodes, rep_nodes):
    reps_text = "\n".join(
        f"{i + 1}. [{r.get('category', '')}] {r.get('name', '')}: {str(r.get('description', ''))[:150]}"
        for i, r in enumerate(rep_nodes)
    )
    type_label = "risk" if node_type == "risk" else "intervention"
    if type_label == "risk":
        task_hint = "For risk sub-clusters: what specific failure mode does this sub-cluster represent?"
    else:
        task_hint = "For intervention sub-clusters: what specific approach does this sub-cluster represent?"

    return [
        {
            "role": "user",
            "content": (
                f"You are naming a sub-cluster in an AI safety knowledge graph.\n"
                f"Parent cluster: {parent_name}\n"
                f"Sub-cluster type: {type_label}\n"
                f"Sub-cluster {sub_id} of 5 (ordered by size, largest first)\n"
                f"N nodes: {n_nodes}\n\n"
                f"Representative nodes (ordered by semantic centrality):\n{reps_text}\n\n"
                f"Task: Generate a concise sub-cluster name and description.\n"
                f"- Name: 5-10 words, captures the specific sub-theme distinct from other sub-clusters\n"
                f"- Description: 1 sentence (max 30 words)\n"
                f"- {task_hint}\n\n"
                'Respond as JSON: {"name": "...", "description": "..."}'
            ),
        }
    ]


def pass2_prompt(node_type, pass1_name, pass1_desc, rep_nodes):
    reps_text = "\n".join(
        f"{i + 1}. [{r.get('category', '')}] {r.get('name', '')}: {str(r.get('description', ''))[:150]}"
        for i, r in enumerate(rep_nodes)
    )
    type_label = "risk" if node_type == "risk" else "intervention"
    return [
        {
            "role": "user",
            "content": (
                f"Review the proposed sub-cluster name for accuracy and specificity.\n"
                f"Sub-cluster type: {type_label}\n"
                f'Proposed name: "{pass1_name}"\n'
                f'Proposed description: "{pass1_desc}"\n\n'
                f"Representative nodes:\n{reps_text}\n\n"
                f"Review criteria:\n"
                f"1. Does the name accurately capture the shared theme? (accurate: true/false)\n"
                f"2. Is the name specific enough to be distinct from other sub-clusters of the same parent? (confidence: high/medium/low)\n"
                f'3. If revisions needed, suggest an improved name (suggested_revision: "..." or null)\n\n'
                'Respond as JSON: {"accurate": bool, "confidence": "high/medium/low", "suggested_revision": "..." or null}'
            ),
        }
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    load_dotenv(os.path.join(PROJECT_ROOT, "graph_analysis/.env"))
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # 1. Load valid_pathway_nodes
    print("Loading valid_pathway_nodes from paths_unconstrained_sim0.9.jsonl ...")
    valid_pathway_nodes = set()
    with open(PATHS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            path = obj.get("path") or obj.get("node_id_sequence") or []
            for nid in path:
                valid_pathway_nodes.add(nid)
    print(f"  valid_pathway_nodes: {len(valid_pathway_nodes):,}")

    # 2. Load cluster_memberships
    print("Loading cluster_memberships.pkl ...")
    with open(CLUSTER_MEMBERSHIPS_PKL, "rb") as f:
        cluster_memberships = pickle.load(f)
    print(f"  Total keys: {len(cluster_memberships):,}")

    # 3. Load subcluster_summary.csv
    print("Loading subcluster_summary.csv ...")
    with open(SUBCLUSTER_SUMMARY_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        subcluster_rows = list(reader)
    print(f"  Rows: {len(subcluster_rows)}")

    # 4. Load node_attributes
    print("Loading graph_node_attributes.pkl ...")
    with open(NODE_ATTRS_PKL, "rb") as f:
        node_attrs = pickle.load(f)
    print(f"  Nodes loaded: {len(node_attrs):,}")

    # Load parent names
    parent_names = load_parent_names()
    print(f"  Parent names loaded: {len(parent_names)}")

    os.makedirs(OUTPUT_DIR_STEP4, exist_ok=True)
    os.makedirs(OUTPUT_DIR_STEP5, exist_ok=True)

    # Output accumulators
    step4_rows = []
    step5_rows = []

    n_parents_processed = 0
    n_subclusters_named = 0
    n_errors = 0

    for row_idx, row in enumerate(subcluster_rows):
        node_type = row["node_type"].strip()
        cluster_id = row["cluster_id"].strip()
        print(
            f"\n[{row_idx + 1}/{len(subcluster_rows)}] Processing {node_type} cluster {cluster_id} "
            f"(n_nodes={row['n_nodes']})"
        )

        # a. Get cluster members
        key = (0.9, "unconstrained", node_type, "agglomerative", cluster_id)
        if key not in cluster_memberships:
            print(f"  WARNING: key {key} not found in cluster_memberships — skipping")
            n_errors += 1
            continue
        members = cluster_memberships[key]
        print(f"  Total cluster members: {len(members)}")

        # b. Filter to valid_pathway_nodes
        qualifying = [m for m in members if m in valid_pathway_nodes]
        print(f"  Qualifying (pathway-filtered) members: {len(qualifying)}")

        # c. Check minimum
        if len(qualifying) < 5:
            print(
                "  Fewer than 5 qualifying members — recording n_subclusters=1, skipping"
            )
            # Record a single placeholder
            parent_name = parent_names.get(
                (node_type, cluster_id), f"{node_type}_{cluster_id}"
            )
            step4_rows.append(
                {
                    "parent_cluster_type": node_type,
                    "parent_cluster_id": cluster_id,
                    "subcluster_id": 0,
                    "n_nodes": len(qualifying),
                    "final_name": f"{parent_name} (undivided)",
                    "llm_description": "Fewer than 5 pathway nodes; no subclustering performed.",
                    "judge_confidence": "n/a",
                    "judge_accurate": "n/a",
                    "top5_node_names": "",
                }
            )
            step5_rows.append(
                {
                    "parent_cluster_type": node_type,
                    "parent_cluster_id": cluster_id,
                    "subcluster_id": 0,
                    "n_nodes": len(qualifying),
                    "final_name": f"{parent_name} (undivided)",
                    "llm_description": "Fewer than 5 pathway nodes; no subclustering performed.",
                    "judge_confidence": "n/a",
                    "judge_accurate": "n/a",
                    "top5_node_names": "",
                    "pass1_name": "",
                    "pass2_revision": "",
                }
            )
            n_parents_processed += 1
            continue

        # d. Parse embeddings
        emb_list = []
        valid_members = []
        for nid in qualifying:
            attrs = node_attrs.get(nid, {})
            emb_raw = attrs.get("embedding")
            if emb_raw is None:
                continue
            try:
                emb = parse_embedding(emb_raw)
                if len(emb) > 0:
                    emb_list.append(emb)
                    valid_members.append(nid)
            except Exception:
                continue

        print(f"  Members with valid embeddings: {len(valid_members)}")
        if len(valid_members) < 5:
            print("  Fewer than 5 valid embeddings — skipping subclustering")
            n_errors += 1
            continue

        emb_matrix = np.stack(emb_list)

        # e. Run AgglomerativeClustering k=5
        print("  Running AgglomerativeClustering k=5 ...")
        n_clusters = min(5, len(valid_members))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(emb_matrix)

        # Build subcluster groups
        sub_groups = {}
        for i, lbl in enumerate(labels):
            sub_groups.setdefault(lbl, []).append(i)

        # Sort subclusters by size descending
        sorted_subs = sorted(sub_groups.keys(), key=lambda lbl: -len(sub_groups[lbl]))

        # Parent name for prompts
        parent_name = parent_names.get(
            (node_type, cluster_id), f"{node_type}_{cluster_id}"
        )
        print(f"  Parent name: {parent_name}")

        n_parents_processed += 1

        # f. Get top-5 representatives per subcluster
        for sub_rank, sub_lbl in enumerate(sorted_subs):
            sub_indices = sub_groups[sub_lbl]
            sub_embs = emb_matrix[sub_indices]
            sub_node_ids = [valid_members[i] for i in sub_indices]

            # Compute mean embedding and cosine sims
            mean_emb = sub_embs.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm

            sims = np.array([cosine_sim(emb_matrix[i], mean_emb) for i in sub_indices])
            top5_idx = np.argsort(-sims)[:5]

            rep_nodes = []
            for idx in top5_idx:
                nid = sub_node_ids[idx]
                attrs = node_attrs.get(nid, {})
                cat = attrs.get("concept_category", attrs.get("type", ""))
                rep_nodes.append(
                    {
                        "name": attrs.get("name", str(nid)),
                        "description": attrs.get("description", ""),
                        "category": cat,
                    }
                )

            n_sub_nodes = len(sub_indices)
            print(
                f"    Subcluster {sub_rank} (label={sub_lbl}): {n_sub_nodes} nodes, "
                f"top5: {[r['name'][:30] for r in rep_nodes]}"
            )

            # g. LLM Pass 1
            p1_messages = pass1_prompt(
                parent_name, node_type, sub_rank, n_sub_nodes, rep_nodes
            )
            p1_result = call_llm(client, p1_messages, max_tokens=200)
            if p1_result is None:
                p1_name = f"{parent_name} subcluster {sub_rank}"
                p1_desc = ""
                print("    Pass 1 FAILED — using fallback name")
                n_errors += 1
            else:
                p1_name = p1_result.get("name", "").strip()
                p1_desc = p1_result.get("description", "").strip()
                print(f"    Pass 1 name: {p1_name}")

            # h. LLM Pass 2 Judge
            p2_messages = pass2_prompt(node_type, p1_name, p1_desc, rep_nodes)
            p2_result = call_llm(client, p2_messages, max_tokens=200)
            if p2_result is None:
                judge_accurate = None
                judge_confidence = "unknown"
                suggested_revision = None
                print("    Pass 2 FAILED")
                n_errors += 1
            else:
                judge_accurate = p2_result.get("accurate", None)
                judge_confidence = p2_result.get("confidence", "unknown")
                suggested_revision = p2_result.get("suggested_revision", None)
                if suggested_revision == "null":
                    suggested_revision = None
                print(
                    f"    Pass 2: accurate={judge_accurate}, confidence={judge_confidence}, "
                    f"revision={suggested_revision}"
                )

            final_name = suggested_revision if suggested_revision else p1_name
            top5_names = "|".join(r["name"] for r in rep_nodes)

            step4_rows.append(
                {
                    "parent_cluster_type": node_type,
                    "parent_cluster_id": cluster_id,
                    "subcluster_id": sub_rank,
                    "n_nodes": n_sub_nodes,
                    "final_name": final_name,
                    "llm_description": p1_desc,
                    "judge_confidence": judge_confidence,
                    "judge_accurate": str(judge_accurate),
                    "top5_node_names": top5_names,
                }
            )
            step5_rows.append(
                {
                    "parent_cluster_type": node_type,
                    "parent_cluster_id": cluster_id,
                    "subcluster_id": sub_rank,
                    "n_nodes": n_sub_nodes,
                    "final_name": final_name,
                    "llm_description": p1_desc,
                    "judge_confidence": judge_confidence,
                    "judge_accurate": str(judge_accurate),
                    "top5_node_names": top5_names,
                    "pass1_name": p1_name,
                    "pass2_revision": suggested_revision if suggested_revision else "",
                }
            )
            n_subclusters_named += 1

        print(f"  Done with cluster {cluster_id}")

    # Save outputs
    step4_fieldnames = [
        "parent_cluster_type",
        "parent_cluster_id",
        "subcluster_id",
        "n_nodes",
        "final_name",
        "llm_description",
        "judge_confidence",
        "judge_accurate",
        "top5_node_names",
    ]
    step5_fieldnames = step4_fieldnames + ["pass1_name", "pass2_revision"]

    with open(OUTPUT_CSV_STEP4, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=step4_fieldnames)
        writer.writeheader()
        writer.writerows(step4_rows)

    with open(OUTPUT_CSV_STEP5, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=step5_fieldnames)
        writer.writeheader()
        writer.writerows(step5_rows)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Parent clusters processed: {n_parents_processed}")
    print(f"  Subclusters named: {n_subclusters_named}")
    print(f"  Errors/warnings: {n_errors}")
    print(f"  Output (step4): {OUTPUT_CSV_STEP4}  ({len(step4_rows)} rows)")
    print(f"  Output (step5): {OUTPUT_CSV_STEP5}  ({len(step5_rows)} rows)")


if __name__ == "__main__":
    main()
