"""
Phase 2 Step 5a — LLM Cluster Naming + Judge Review

Three-pass workflow:
  Pass 1: LLM drafts synthesized names for all ~120 clusters (risk + intervention + chain)
  Pass 2: LLM-as-judge reviews each name, flags split candidates and low-confidence names
  Pass 3: Outputs human review checklist (mandatory + auto-flagged)

Inputs:
  cluster_memberships.pkl, graph_node_attributes.pkl
  optionA_cluster_labels.pkl  (body node IDs per chain cluster)
  risk_clusters.csv, intervention_clusters.csv (for n_nodes/n_sources reference)

Outputs (step5_naming/):
  risk_cluster_names_llm.csv
  intervention_cluster_names_llm.csv
  chain_cluster_names_llm.csv
  all_clusters_naming_detail.csv
  human_review_checklist.csv
"""

import json
import pickle
import logging
import csv
import time
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
# Load .env from graph_analysis/ dir; key may be lowercase so normalise
_env_path = BASE / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
    # Normalise lowercase key to uppercase for OpenAI client
    import os as _os

    if not _os.environ.get("OPENAI_API_KEY") and _os.environ.get("openai_api_key"):
        _os.environ["OPENAI_API_KEY"] = _os.environ["openai_api_key"]
PKL_DIR = BASE / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4_DIR = BASE / "phase2_results/step4_finalanalysis"
OUT_DIR = BASE / "phase2_results/step5_naming"
LOG_DIR = BASE.parent / "logfiles/phase5_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase2_step5_naming.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── OpenAI client ────────────────────────────────────────────────────────────
client = OpenAI()
MODEL = "gpt-5.4-mini"

# ── Load PKLs ─────────────────────────────────────────────────────────────────
log.info("Loading PKL files …")
t0 = time.time()
with open(PKL_DIR / "cluster_memberships.pkl", "rb") as f:
    cluster_memberships = pickle.load(f)
log.info(
    f"  cluster_memberships: {len(cluster_memberships)} keys  ({time.time() - t0:.1f}s)"
)

t0 = time.time()
with open(PKL_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
log.info(f"  node_attrs: {len(node_attrs)} nodes  ({time.time() - t0:.1f}s)")

# ── Load valid_pathway_nodes (Gap 5a fix) ─────────────────────────────────────
# valid_pathway_nodes = nodes appearing on any qualifying path (EDGE conf≥3,
# SIM cos_sim≥0.9, maturity≥3, applied simultaneously during path generation).
# Using unconstrained VPN as authoritative qualifying universe for naming.
PATHS_UNCONSTRAINED = (
    BASE.parent / "graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"
)
if not PATHS_UNCONSTRAINED.exists():
    # Try relative path for scripts run from project root
    PATHS_UNCONSTRAINED = BASE / "phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"

log.info("Building valid_pathway_nodes from unconstrained path file …")
t0 = time.time()
valid_pathway_nodes = set()
with open(PATHS_UNCONSTRAINED) as _f:
    for _line in _f:
        _obj = json.loads(_line)
        _path = _obj.get("path") or _obj.get("node_id_sequence") or []
        for _nid in _path:
            valid_pathway_nodes.add(int(_nid))
log.info(
    f"  valid_pathway_nodes: {len(valid_pathway_nodes):,} nodes  ({time.time() - t0:.1f}s)"
)

# optionA cluster labels: dict cluster_id -> set of body node IDs
LABELS_PKL = STEP4_DIR / "optionA_cluster_labels.pkl"
if LABELS_PKL.exists():
    with open(LABELS_PKL, "rb") as f:
        optionA_labels = pickle.load(f)
    log.info(f"  optionA_cluster_labels: {len(optionA_labels)} clusters")
else:
    optionA_labels = None
    log.warning(
        "  optionA_cluster_labels.pkl not found — will derive chain members from KMeans model + path file"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    norm = np.linalg.norm(c)
    return c / norm if norm > 0 else c


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


def format_nodes(node_ids):
    lines = []
    for i, nid in enumerate(node_ids, 1):
        attrs = node_attrs.get(nid, {})
        name = attrs.get("name", f"[node {nid}]")
        desc = (attrs.get("description") or "")[:150]
        cat = attrs.get("concept_category") or attrs.get("type", "")
        lines.append(f"{i}. [{cat}] {name}: {desc}")
    return "\n".join(lines)


def get_cluster_dict(node_type):
    """Return {cluster_id: [node_ids]} for a given node_type at SIM0.9/unconstrained/agglomerative.
    edge_config is stored as float 0.9 in the pkl, not string '0.9'.
    Gap 5a fix: members filtered to valid_pathway_nodes (unconstrained qualifying universe).
    """
    result = {}
    for (ec, mode, nt, algo, cid), members in cluster_memberships.items():
        if (
            ec == 0.9
            and mode == "unconstrained"
            and nt == node_type
            and algo == "agglomerative"
        ):
            filtered = [nid for nid in members if nid in valid_pathway_nodes]
            if filtered:
                result[int(cid)] = filtered
    return result


def build_chain_cluster_dict():
    """
    Build chain cluster dict: {cluster_id: [node_ids]} from consim1 paths (selected config).
    optionA_labels was built from consim2 paths; for the selected config (consim1), always
    derive from the consim1 path file to ensure correct config alignment.
    Body nodes filtered to valid_pathway_nodes (Gap 5a fix).
    """
    # Always derive from consim1 path file (selected config) for correct config alignment.
    # optionA_labels is from consim2 and is NOT used for the selected config.
    return derive_chain_clusters_from_paths()

    # Legacy code below (unreachable) kept for reference:
    if optionA_labels is None:
        return derive_chain_clusters_from_paths()

    # Actual format: dict with keys "labels" and "records"
    if (
        isinstance(optionA_labels, dict)
        and "labels" in optionA_labels
        and "records" in optionA_labels
    ):
        labels_arr = optionA_labels["labels"]  # shape (N,) int cluster IDs
        records = optionA_labels["records"]  # list of (body_ids, full_path_ids)
        result = {}
        for lab, rec in zip(labels_arr, records):
            cid = int(lab)
            body_ids = (
                rec[0] if isinstance(rec, (list, tuple)) and len(rec) >= 1 else []
            )
            for nid in body_ids:
                if nid in valid_pathway_nodes:
                    result.setdefault(cid, []).append(nid)
        log.info(
            f"  Chain clusters from optionA_labels: {len(result)} clusters, "
            f"{sum(len(v) for v in result.values())} total body nodes"
        )
        return result

    # Legacy: {cluster_id: set/list of node_ids}
    if isinstance(optionA_labels, dict):
        first_val = next(iter(optionA_labels.values()))
        if isinstance(first_val, (set, list, np.ndarray)):
            return {int(k): list(v) for k, v in optionA_labels.items()}

    log.warning("Unrecognised optionA_labels format — falling back to path derivation")
    return derive_chain_clusters_from_paths()


def derive_chain_clusters_from_paths():
    """Derive chain body node sets by streaming consim1 path file and running KMeans predict.
    Uses consim1 (selected config). Body nodes filtered to valid_pathway_nodes."""
    import pickle as pkl

    model_path = STEP4_DIR / "optionA_kmeans_model.pkl"
    # Use selected config (consim1) path file
    paths_file = STEP4_DIR / "step4_paths/representative_pathways_consim1.jsonl"
    if not model_path.exists() or not paths_file.exists():
        log.error("Cannot derive chain clusters: KMeans model or path file missing")
        return {}
    with open(model_path, "rb") as f:
        kmeans = pkl.load(f)
    log.info("Deriving chain cluster body node sets from path file + KMeans predict …")
    result = {}
    BATCH = 2000
    batch_embs, batch_bids = [], []

    def flush():
        if not batch_embs:
            return
        labels = kmeans.predict(np.stack(batch_embs))
        for lab, bids in zip(labels, batch_bids):
            result.setdefault(int(lab), set()).update(bids)
        batch_embs.clear()
        batch_bids.clear()

    with open(paths_file) as f:
        for line in f:
            obj = json.loads(line)
            path = [int(x) for x in obj["node_id_sequence"]]
            body = path[1:-1]
            embs_b = [
                get_embedding(nid) for nid in body if get_embedding(nid) is not None
            ]
            if not embs_b:
                continue
            mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
            batch_embs.append(mean_emb)
            batch_bids.append([nid for nid in body if nid in valid_pathway_nodes])
            if len(batch_embs) >= BATCH:
                flush()
    flush()
    return {k: list(v) for k, v in result.items()}


# ── LLM prompts ───────────────────────────────────────────────────────────────
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
        "What conceptual bridge (intermediate reasoning) connects a risk to an intervention? "
        "Frame as: 'the body of thinking about [X] that connects [risk type] to [intervention type]'."
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
        f"1. Does the name accurately capture the shared theme across MOST of these nodes? (accurate)\n"
        f"2. Do the nodes clearly span more than 2 distinct themes? (split_candidate)\n"
        f"3. Is the name specific enough to distinguish from adjacent {cluster_type} clusters? (confidence)\n"
        f"4. If revisions are needed, suggest an improved name (suggested_revision).\n\n"
        f"Respond as valid JSON only:\n"
        f'{{"accurate": true/false, "issues": "brief description or null", '
        f'"split_candidate": true/false, "confidence": "high/medium/low", '
        f'"suggested_revision": "revised name or null"}}'
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
            log.warning(f"  LLM call failed (attempt {attempt + 1}): {e}")
            time.sleep(2**attempt)
    return {}


# ── Process one cluster type ───────────────────────────────────────────────────
def process_clusters(cluster_type, cluster_dict):
    rows = []
    total = len(cluster_dict)
    for i, cid in enumerate(sorted(cluster_dict.keys()), 1):
        members = cluster_dict[cid]
        centroid = compute_centroid(members)
        if centroid is None:
            log.warning(f"  {cluster_type} cluster {cid}: no embeddings, skipping")
            continue
        top_ids = top_n_by_centroid_sim(members, centroid, n=10)
        nodes_text = format_nodes(top_ids)
        top5_names = " | ".join(
            node_attrs.get(nid, {}).get("name", "") for nid in top_ids[:5]
        )

        # Pass 1: naming
        p1 = call_llm(naming_prompt(cluster_type, nodes_text))
        llm_name = p1.get("name", "")
        llm_desc = p1.get("description", "")

        # Pass 2: judge
        p2 = call_llm(judge_prompt(cluster_type, nodes_text, llm_name, llm_desc))
        accurate = p2.get("accurate")
        issues = p2.get("issues")
        split_cand = bool(p2.get("split_candidate", False))
        confidence = p2.get("confidence", "")
        suggestion = p2.get("suggested_revision")
        final_name = suggestion if suggestion and suggestion != "null" else llm_name

        rows.append(
            {
                "cluster_type": cluster_type,
                "cluster_id": cid,
                "n_members": len(members),
                "llm_name": llm_name,
                "llm_description": llm_desc,
                "judge_accurate": accurate,
                "judge_issues": issues or "",
                "judge_split_candidate": split_cand,
                "judge_confidence": confidence,
                "suggested_revision": suggestion or "",
                "final_name": final_name,
                "top5_node_names": top5_names,
            }
        )
        log.info(
            f"  [{cluster_type} {i}/{total}] cluster {cid}: "
            f"'{final_name[:60]}' [conf={confidence}, split={split_cand}]"
        )
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("Phase 2 Step 5 — LLM Cluster Naming")
    log.info("=" * 70)

    all_rows = []

    # Risk clusters
    log.info("\n── Risk clusters ──")
    risk_dict = get_cluster_dict("risk")
    log.info(f"  {len(risk_dict)} clusters")
    risk_rows = process_clusters("risk", risk_dict)
    all_rows.extend(risk_rows)

    # Intervention clusters
    log.info("\n── Intervention clusters ──")
    interv_dict = get_cluster_dict("intervention")
    log.info(f"  {len(interv_dict)} clusters")
    interv_rows = process_clusters("intervention", interv_dict)
    all_rows.extend(interv_rows)

    # Chain clusters (Option A)
    log.info("\n── Chain body clusters (Option A) ──")
    chain_dict = build_chain_cluster_dict()
    log.info(
        f"  {len(chain_dict)} clusters, total body nodes: {sum(len(v) for v in chain_dict.values())}"
    )
    chain_rows = process_clusters("chain", chain_dict)
    all_rows.extend(chain_rows)

    # ── Write output CSVs ────────────────────────────────────────────────────
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

    def write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        log.info(f"  Saved {path}  ({len(rows)} rows)")

    write_csv(OUT_DIR / "all_clusters_naming_detail.csv", all_rows)
    write_csv(OUT_DIR / "risk_cluster_names_llm.csv", risk_rows)
    write_csv(OUT_DIR / "intervention_cluster_names_llm.csv", interv_rows)
    write_csv(OUT_DIR / "chain_cluster_names_llm.csv", chain_rows)

    # ── Human review checklist ───────────────────────────────────────────────
    set(range(40))  # all 40 risk clusters mandatory
    checklist = []
    for row in all_rows:
        mandatory = (
            row["cluster_type"] == "risk"
            or not row["judge_accurate"]
            or row["judge_confidence"] in ("medium", "low")
            or row["judge_split_candidate"]
        )
        if mandatory:
            checklist.append(
                {
                    "cluster_type": row["cluster_type"],
                    "cluster_id": row["cluster_id"],
                    "final_name": row["final_name"],
                    "review_reason": (
                        "mandatory_risk"
                        if row["cluster_type"] == "risk"
                        else "judge_inaccurate"
                        if not row["judge_accurate"]
                        else "split_candidate"
                        if row["judge_split_candidate"]
                        else f"low_confidence_{row['judge_confidence']}"
                    ),
                    "judge_issues": row["judge_issues"],
                    "judge_confidence": row["judge_confidence"],
                    "top5_node_names": row["top5_node_names"],
                }
            )

    checklist_fields = [
        "cluster_type",
        "cluster_id",
        "final_name",
        "review_reason",
        "judge_issues",
        "judge_confidence",
        "top5_node_names",
    ]
    with open(
        OUT_DIR / "human_review_checklist.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=checklist_fields)
        w.writeheader()
        w.writerows(checklist)
    log.info(f"  Saved human_review_checklist.csv  ({len(checklist)} clusters flagged)")

    log.info("\n── Summary ──")
    log.info(f"  Total clusters named: {len(all_rows)}")
    high_conf = sum(1 for r in all_rows if r["judge_confidence"] == "high")
    splits = sum(1 for r in all_rows if r["judge_split_candidate"])
    inaccurate = sum(1 for r in all_rows if not r["judge_accurate"])
    log.info(f"  High confidence: {high_conf}/{len(all_rows)}")
    log.info(f"  Split candidates: {splits}")
    log.info(f"  Judge-inaccurate: {inaccurate}")
    log.info(f"  Human review checklist: {len(checklist)} clusters")
    log.info("Done.")


if __name__ == "__main__":
    main()
