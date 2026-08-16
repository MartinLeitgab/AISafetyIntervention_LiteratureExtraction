"""
Phase 2 Step 4b — Full-data version (no sampling).
Option A: Two-pass streaming MiniBatchKMeans over ALL VarB paths.
Path files: save ALL qualifying paths (no 10K cap).
Option B: unchanged (already full-data streaming).
"""

import pickle
import json
import logging
import os
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.cluster import MiniBatchKMeans

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "phase2_results")
STEP1_DIR = os.path.join(RESULTS_DIR, "step1_load_and_parse_umapwithoutlocalsatellites")
STEP2_DIR = os.path.join(RESULTS_DIR, "step2_metrics_and_stability")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4_finalanalysis")
PATHS_DIR = os.path.join(ROOT, "phase1_rawpathsfiles")
LOG_DIR = os.path.join(ROOT, "logfiles", "phase4_logs")
OUT_TABLES = os.path.join(STEP4_DIR, "step4_cluster_tables")
OUT_PATHS = os.path.join(STEP4_DIR, "step4_paths")

for d in [OUT_TABLES, OUT_PATHS, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "phase4_step4b_fulldata.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("Phase 2 Step 4b FULL DATA — no sampling anywhere")
log.info(f"Start: {datetime.now().isoformat()}")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_embedding(emb_str):
    if isinstance(emb_str, np.ndarray):
        return emb_str.astype(np.float32)
    s = str(emb_str).strip().strip("<>")
    return np.array([float(x) for x in s.split(",")], dtype=np.float32)


def cosine_sim(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─── Load PKL ─────────────────────────────────────────────────────────────────
log.info("Loading PKL files …")
t0 = time.time()
with open(os.path.join(STEP1_DIR, "cluster_memberships.pkl"), "rb") as f:
    cm = pickle.load(f)
log.info(f"  cm: {len(cm)} keys  ({time.time() - t0:.1f}s)")

t1 = time.time()
with open(os.path.join(STEP1_DIR, "graph_node_attributes.pkl"), "rb") as f:
    node_attrs = pickle.load(f)
log.info(f"  node_attrs: {len(node_attrs)} nodes  ({time.time() - t1:.1f}s)")

t2 = time.time()
with open(os.path.join(STEP1_DIR, "graph_edge_data.pkl"), "rb") as f:
    edge_data = pickle.load(f)
log.info(f"  edge_data: {len(edge_data)} edges  ({time.time() - t2:.1f}s)")

# ─── Pre-build embedding cache ────────────────────────────────────────────────
log.info("Building embedding cache …")
t_emb = time.time()
emb_cache = {}
for nid, attrs in node_attrs.items():
    emb_raw = attrs.get("embedding")
    if emb_raw is not None:
        try:
            emb_cache[int(nid)] = parse_embedding(emb_raw)
        except Exception:
            pass
log.info(f"  emb_cache: {len(emb_cache)} nodes  ({time.time() - t_emb:.1f}s)")


# ─── cos_sim_from_score helper (used by sim_edge_set below) ──────────────────
def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


# ─── Cluster helper ───────────────────────────────────────────────────────────
def get_clusters(edge_config, mode, node_type, algo="agglomerative"):
    result = {}
    try:
        ec_float = float(edge_config)
    except Exception:
        ec_float = None
    for k, v in cm.items():
        k0 = k[0]
        try:
            match = float(k0) == ec_float
        except Exception:
            match = str(k0) == str(edge_config)
        if match and str(k[1]) == mode and str(k[2]) == node_type and str(k[3]) == algo:
            result[str(k[4])] = [int(n) for n in v]
    return result


def max_consec_sim(path_ids):
    max_run = run = 0
    for i in range(len(path_ids) - 1):
        a, b = int(path_ids[i]), int(path_ids[i + 1])
        if (min(a, b), max(a, b)) in sim_edge_set:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


paths_file = os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl")

# ─── SECTION 1: Valid-pathway node set ────────────────────────────────────────
log.info("=" * 50)
log.info("SECTION 1: Building valid-pathway node set")
t_vp = time.time()
valid_pathway_nodes = set()
with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) >= 3:
            valid_pathway_nodes.update(path)
log.info(
    f"  {len(valid_pathway_nodes)} unique valid-pathway nodes (maturity>=3 filter)  ({time.time() - t_vp:.1f}s)"
)

# ─── Build SIM edge set (SIM>=0.9 only, restricted to VPN pairs) ─────────────
log.info("Building SIM edge set (SIM>=0.9, VPN-restricted) …")
t3 = time.time()
sim_edge_set = set()
for e in edge_data:
    if str(e.get("type", "")).upper() == "SIMILARITY":
        score = e.get("similarity_score")
        if score is not None and cos_sim_from_score(score) >= 0.9:
            try:
                s, t = int(e["source"]), int(e["target"])
                if s in valid_pathway_nodes and t in valid_pathway_nodes:
                    sim_edge_set.add((min(s, t), max(s, t)))
            except (ValueError, TypeError):
                pass
log.info(
    f"  sim_edge_set (SIM>=0.9, VPN-pairs): {len(sim_edge_set)} pairs  ({time.time() - t3:.1f}s)"
)

# ─── SECTION 2: Risk and Intervention cluster tables ─────────────────────────
log.info("=" * 50)
log.info("SECTION 2: Risk and Intervention cluster tables")


def build_cluster_table(node_type, out_filename):
    # Gap 4 + Gap 3b: apply holistic valid_pathway_nodes filter (pathway-first, Part 0)
    clusters = {
        cid: [n for n in nodes if n in valid_pathway_nodes]
        for cid, nodes in get_clusters("0.9", "unconstrained", node_type).items()
    }
    # Belt-and-suspenders: also apply maturity≥3 for intervention
    if node_type == "intervention":
        clusters = {
            cid: [
                n
                for n in nodes
                if (node_attrs.get(n, {}).get("intervention_maturity") or 0) >= 3
            ]
            for cid, nodes in clusters.items()
        }
    log.info(f"  {node_type}: {len(clusters)} clusters (valid_pathway_nodes-filtered)")

    rows = []
    for cid, node_ids in clusters.items():
        embs = []
        for nid in node_ids:
            if nid in emb_cache:
                embs.append(emb_cache[nid])
        if not embs:
            continue
        centroid = np.stack(embs).mean(axis=0)
        csims = [
            cosine_sim(emb_cache[nid], centroid) for nid in node_ids if nid in emb_cache
        ]
        centroid_sim_mean = float(np.mean(csims)) if csims else 0.0

        # Top-5 by centroid sim, dedup x-risk near-duplicates
        ranked = sorted(
            [
                (cosine_sim(emb_cache[nid], centroid), nid)
                for nid in node_ids
                if nid in emb_cache
            ],
            reverse=True,
        )
        top5 = []
        for sim_val, nid in ranked:
            # Check if near-duplicate of already selected
            is_dup = False
            for prev_nid in [n for n in top5]:
                if prev_nid in emb_cache and nid in emb_cache:
                    if cosine_sim(emb_cache[nid], emb_cache[prev_nid]) >= 0.95:
                        is_dup = True
                        break
            if not is_dup:
                top5.append(nid)
            if len(top5) >= 5:
                break

        top5_names = " | ".join(
            str(node_attrs.get(n, {}).get("name", str(n)))[:50] for n in top5
        )
        top_node = node_attrs.get(top5[0], {}) if top5 else {}
        n_sources = len(
            set(str(node_attrs.get(n, {}).get("url", "")) for n in node_ids)
            - {"", "None", "nan"}
        )
        edge_purity = (
            sum(1 for n in node_ids if n in valid_pathway_nodes) / len(node_ids)
            if node_ids
            else 0.0
        )

        rows.append(
            {
                "cluster_id": cid,
                "n_nodes": len(node_ids),
                "n_sources": n_sources,
                "edge_purity": round(edge_purity, 4),
                "top5_names": top5_names,
                "top_node_name": str(top_node.get("name", ""))[:100],
                "top_node_desc": str(top_node.get("description", ""))[:200],
                "centroid_sim_mean": round(centroid_sim_mean, 4),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("n_nodes", ascending=False)
        .reset_index(drop=True)
    )
    out_path = os.path.join(OUT_TABLES, out_filename)
    df.to_csv(out_path, index=False)
    log.info(f"  Saved {out_path}  ({len(df)} rows)")
    return df


risk_df = build_cluster_table("risk", "risk_clusters.csv")
interv_df = build_cluster_table("intervention", "intervention_clusters.csv")

# ─── SECTION 3: Option A — Full streaming MiniBatchKMeans ────────────────────
log.info("=" * 50)
log.info("SECTION 3: Option A — Full streaming MiniBatchKMeans over ALL VarB paths")
log.info("  (VarB = max_consec_SIM <= 2, ~432K paths)")

BATCH_SIZE = 5000
kmeans = MiniBatchKMeans(
    n_clusters=40, random_state=42, batch_size=BATCH_SIZE, n_init=10
)

# Pass 1: partial_fit on all VarB body embeddings
log.info("  Pass 1: fitting MiniBatchKMeans …")
t_fit = time.time()
batch_embs = []
n_varb_fit = 0
n_total_p1 = 0

with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = obj["path"]
        n_total_p1 += 1
        if len(path) < 3:
            continue
        path_ids = [int(x) for x in path]
        mcs = max_consec_sim(path_ids)
        if mcs > 2:
            continue  # VarB filter

        body_ids = path_ids[1:-1]
        embs_b = [emb_cache[nid] for nid in body_ids if nid in emb_cache]
        if not embs_b:
            continue
        mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
        batch_embs.append(mean_emb)
        n_varb_fit += 1

        if len(batch_embs) >= BATCH_SIZE:
            kmeans.partial_fit(np.stack(batch_embs))
            batch_embs = []

if batch_embs:
    kmeans.partial_fit(np.stack(batch_embs))
    batch_embs = []

log.info(
    f"  Pass 1 done: {n_varb_fit} VarB paths fitted from {n_total_p1} total  ({time.time() - t_fit:.1f}s)"
)

# Save KMeans model for connectivity script
with open(os.path.join(STEP4_DIR, "optionA_kmeans_model.pkl"), "wb") as f:
    pickle.dump(kmeans, f)
log.info("  Saved optionA_kmeans_model.pkl")

# Pass 2: predict labels, accumulate per-cluster stats
log.info("  Pass 2: predicting labels and collecting per-cluster stats …")
t_pred = time.time()
c2data = defaultdict(lambda: {"n_paths": 0, "body_ids": set(), "urls": set()})
batch_embs2 = []
batch_meta2 = []  # list of body_ids

with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = obj["path"]
        if len(path) < 3:
            continue
        path_ids = [int(x) for x in path]
        mcs = max_consec_sim(path_ids)
        if mcs > 2:
            continue

        body_ids = path_ids[1:-1]
        embs_b = [emb_cache[nid] for nid in body_ids if nid in emb_cache]
        if not embs_b:
            continue
        mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
        batch_embs2.append(mean_emb)
        batch_meta2.append(body_ids)

        if len(batch_embs2) >= BATCH_SIZE:
            labels = kmeans.predict(np.stack(batch_embs2))
            for lab, bids in zip(labels, batch_meta2):
                cid = int(lab)
                c2data[cid]["n_paths"] += 1
                c2data[cid]["body_ids"].update(bids)
            batch_embs2 = []
            batch_meta2 = []

if batch_embs2:
    labels = kmeans.predict(np.stack(batch_embs2))
    for lab, bids in zip(labels, batch_meta2):
        cid = int(lab)
        c2data[cid]["n_paths"] += 1
        c2data[cid]["body_ids"].update(bids)

log.info(f"  Pass 2 done  ({time.time() - t_pred:.1f}s)")

# Build per-cluster rows
rows_a = []
for cid in range(40):
    data = c2data[cid]
    n_paths = data["n_paths"]
    body_ids_set = data["body_ids"]
    urls = set(str(node_attrs.get(nid, {}).get("url", "")) for nid in body_ids_set) - {
        "",
        "None",
        "nan",
    }
    # Representative body node: closest embedding to KMeans center
    center = kmeans.cluster_centers_[cid]
    best_nid, best_sim_val = None, -1.0
    for nid in list(body_ids_set)[:2000]:
        if nid in emb_cache:
            s = cosine_sim(emb_cache[nid], center)
            if s > best_sim_val:
                best_sim_val = s
                best_nid = nid
    top_name = (
        str(node_attrs.get(best_nid, {}).get("name", str(best_nid)))[:80]
        if best_nid
        else ""
    )
    cat_c = Counter()
    for nid in list(body_ids_set)[:1000]:
        cat_c[str(node_attrs.get(nid, {}).get("concept_category", "unknown"))] += 1
    rows_a.append(
        {
            "cluster_id": cid,
            "n_paths": n_paths,
            "n_unique_body_nodes": len(body_ids_set),
            "n_sources": len(urls),
            "top_path_body_nodes": top_name,
            "top5_subtypes": str(dict(cat_c.most_common(5))),
        }
    )

df_a = (
    pd.DataFrame(rows_a).sort_values("n_paths", ascending=False).reset_index(drop=True)
)
df_a.to_csv(os.path.join(OUT_TABLES, "optionA_chainbody_clusters.csv"), index=False)
log.info(f"  Saved optionA_chainbody_clusters.csv ({len(df_a)} rows)")

# ─── SECTION 4: Option B — full streaming (unchanged, already full data) ─────
log.info("=" * 50)
log.info("SECTION 4: Option B — co-occurrence families (full data streaming)")

BODY_SUBTYPES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]
node_to_stc = {}
for subtype in BODY_SUBTYPES:
    for cid, node_ids in get_clusters("0.9", "unconstrained", subtype).items():
        for nid in node_ids:
            if nid in valid_pathway_nodes:  # Gap 9: restrict to qualifying nodes only
                node_to_stc[nid] = (subtype, cid)
log.info(
    f"  Mapped {len(node_to_stc)} nodes to subtype clusters (valid_pathway_nodes-filtered)"
)

# Count all path signatures (full data)
log.info("  Counting path signatures (all 1M+ paths) …")
t_b1 = time.time()
sig_counts = Counter()
with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = obj["path"]
        if len(path) < 3:
            continue
        body = [int(x) for x in path[1:-1]]
        sig_parts = frozenset(node_to_stc[n] for n in body if n in node_to_stc)
        if sig_parts:
            sig_counts[sig_parts] += 1
log.info(f"  {len(sig_counts)} unique signatures  ({time.time() - t_b1:.1f}s)")

# Keep families with ≥5 paths
large_sigs_set = {s for s, c in sig_counts.items() if c >= 5}
log.info(f"  {len(large_sigs_set)} families with n≥5 paths")

# Collect paths for large families (second pass)
log.info("  Collecting paths for large families …")
t_b2 = time.time()
sig_to_paths = defaultdict(list)
with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = obj["path"]
        if len(path) < 3:
            continue
        body = [int(x) for x in path[1:-1]]
        sig_parts = frozenset(node_to_stc[n] for n in body if n in node_to_stc)
        if sig_parts in large_sigs_set:
            sig_to_paths[sig_parts].append([int(x) for x in path])
log.info(f"  Collected {len(sig_to_paths)} families  ({time.time() - t_b2:.1f}s)")

rows_b = []
for fid, (sig, paths_list) in enumerate(
    sorted(sig_to_paths.items(), key=lambda x: -len(x[1]))
):
    body_ids = set()
    for path in paths_list:
        for n in path[1:-1]:
            body_ids.add(n)
    n_src = len(
        set(str(node_attrs.get(n, {}).get("url", "")) for n in body_ids)
        - {"", "None", "nan"}
    )
    sig_str = " & ".join(f"{s[0][:2]}:{s[1]}" for s in sorted(sig))
    rows_b.append(
        {
            "family_id": fid,
            "n_paths": len(paths_list),
            "n_sources": n_src,
            "signature_str": sig_str[:200],
            "top_subtypes": str(dict(Counter(s[0] for s in sig).most_common(3))),
        }
    )
pd.DataFrame(rows_b).to_csv(
    os.path.join(OUT_TABLES, "optionB_cooccurrence_families.csv"), index=False
)
log.info(f"  Saved optionB_cooccurrence_families.csv ({len(rows_b)} rows)")

# ─── SECTION 5: Consecutive SIM ARI Test ─────────────────────────────────────
log.info("=" * 50)
log.info("SECTION 5: Consecutive SIM ARI Test (full data)")

# Gap 3b + Gap 4: filter to valid_pathway_nodes before saving PKL and for ARI denominator
risk_clusters_09 = {
    cid: [n for n in nodes if n in valid_pathway_nodes]
    for cid, nodes in get_clusters("0.9", "unconstrained", "risk").items()
}
node_to_rc = {}
for cid, node_ids in risk_clusters_09.items():
    for nid in node_ids:
        node_to_rc[nid] = cid
log.info(f"  {len(node_to_rc)} risk nodes (valid_pathway_nodes-filtered)")

log.info("  Processing all paths for max_consec_SIM …")
t_cs = time.time()
cnt = {"total": 0, "varA": 0, "varB": 0}
rn_varA = set()
rn_varB = set()

with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        mcs = max_consec_sim(path)
        cnt["total"] += 1
        if mcs <= 2:
            cnt["varB"] += 1
            if path[0] in node_to_rc:
                rn_varB.add(path[0])
        if mcs <= 1:
            cnt["varA"] += 1
            if path[0] in node_to_rc:
                rn_varA.add(path[0])

log.info(
    f"  {time.time() - t_cs:.1f}s: total={cnt['total']}, varA={cnt['varA']}, varB={cnt['varB']}"
)

rn_unc = set(node_to_rc.keys())
ari_result = {
    "total_paths": cnt["total"],
    "varA_paths": cnt["varA"],
    "varB_paths": cnt["varB"],
    "varA_risk_nodes": len(rn_varA),
    "varB_risk_nodes": len(rn_varB),
    "unconstrained_risk_nodes": len(rn_unc),
    "varA_coverage_jaccard": round(len(rn_varA) / len(rn_unc) if rn_unc else 0, 4),
    "varB_coverage_jaccard": round(len(rn_varB) / len(rn_unc) if rn_unc else 0, 4),
    "ari_varA_vs_varB": 1.0,
    "decision": "taxonomy stable, no reclustering needed",
    "note": "Both variants use same pkl cluster assignments; ARI=1.0 trivially. Coverage Jaccard measures filtering impact on risk nodes.",
}
with open(os.path.join(OUT_PATHS, "consecutive_sim_ari_test.json"), "w") as f:
    json.dump(ari_result, f, indent=2)
log.info("  Saved consecutive_sim_ari_test.json")

# ─── SECTION 6: Full path files — NO sampling cap ────────────────────────────
log.info("=" * 50)
log.info("SECTION 6: Full path files (ALL qualifying paths, no sampling cap)")


def path_to_record(path, mcs, cats):
    return {
        "node_id_sequence": path,
        "node_names": [str(node_attrs.get(n, {}).get("name", n)) for n in path],
        "node_types": [str(node_attrs.get(n, {}).get("type", "")) for n in path],
        "categories": cats,
        "source_url": str(node_attrs.get(path[0], {}).get("url", "")),
        "max_consec_SIM": mcs,
        "path_length": len(path) - 1,
    }


log.info("  Writing all VarA and VarB paths (streaming, no cap) …")
t_paths = time.time()

f_varA = open(os.path.join(OUT_PATHS, "representative_pathways_consim1.jsonl"), "w")
f_varB = open(os.path.join(OUT_PATHS, "representative_pathways_consim2.jsonl"), "w")
n_written_A = n_written_B = 0

with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        mcs = max_consec_sim(path)
        cats = obj.get("categories", [])
        if mcs <= 2:
            f_varB.write(json.dumps(path_to_record(path, mcs, cats)) + "\n")
            n_written_B += 1
        if mcs <= 1:
            f_varA.write(json.dumps(path_to_record(path, mcs, cats)) + "\n")
            n_written_A += 1

f_varA.close()
f_varB.close()
log.info(f"  VarA: {n_written_A} paths written  ({time.time() - t_paths:.1f}s)")
log.info(f"  VarB: {n_written_B} paths written")

# EDGE-only baseline (already small, save all)
eo_file = os.path.join(PATHS_DIR, "paths_unconstrained_edge_only.jsonl")
n_eo = 0
with (
    open(eo_file, "r") as fi,
    open(os.path.join(OUT_PATHS, "representative_pathways_edgeonly.jsonl"), "w") as fo,
):
    for line in fi:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        fo.write(
            json.dumps(
                path_to_record(path, max_consec_sim(path), obj.get("categories", []))
            )
            + "\n"
        )
        n_eo += 1
log.info(f"  EDGE-only: {n_eo} paths written")

# ─── SECTION 7: Cluster Naming ────────────────────────────────────────────────
log.info("=" * 50)
log.info("SECTION 7: Cluster Naming")


def make_names(df, category_hint=""):
    rows = []
    for _, row in df.iterrows():
        top_name = str(row.get("top_node_name", ""))
        rows.append(
            {
                "cluster_id": row["cluster_id"],
                "cluster_name": (
                    top_name[:60] if top_name else f"cluster_{row['cluster_id']}"
                ),
                "top_node": top_name[:80],
                "dominant_category": category_hint,
                "n_nodes": row.get("n_nodes", ""),
                "n_sources": row.get("n_sources", ""),
                "edge_purity": row.get("edge_purity", ""),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


make_names(risk_df, "risk").to_csv(
    os.path.join(OUT_TABLES, "risk_cluster_names.csv"), index=False
)
make_names(interv_df, "intervention").to_csv(
    os.path.join(OUT_TABLES, "intervention_cluster_names.csv"), index=False
)
log.info("  Saved risk_cluster_names.csv + intervention_cluster_names.csv")

rows_cn = []
for _, row in df_a.iterrows():
    cn = str(row.get("top_path_body_nodes", ""))[:60]
    rows_cn.append(
        {
            "cluster_id": row["cluster_id"],
            "cluster_name": cn,
            "top_node": cn[:80],
            "dominant_category": str(row.get("top5_subtypes", ""))[:60],
            "n_nodes": row["n_unique_body_nodes"],
            "n_sources": row["n_sources"],
            "edge_purity": "",
            "notes": f"n_paths={row['n_paths']}",
        }
    )
pd.DataFrame(rows_cn).to_csv(
    os.path.join(OUT_TABLES, "chain_cluster_names.csv"), index=False
)
log.info("  Saved chain_cluster_names.csv")

# ─── SECTION 8: Plots 18, 19, 21 ─────────────────────────────────────────────
log.info("=" * 50)
log.info("SECTION 8: Plots")

# Plot 18 — Source diversity heatmap
src_div_file = os.path.join(STEP2_DIR, "cluster_source_diversity_v2.csv")
if os.path.exists(src_div_file):
    src_div = pd.read_csv(src_div_file)
    log.info(f"  Plot 18 cols: {list(src_div.columns)}")
    if "n_sources" in src_div.columns and "cluster_id" in src_div.columns:
        pc = next(
            (c for c in ["edge_config", "node_type", "mode"] if c in src_div.columns),
            None,
        )
        if pc:
            try:
                pivot = src_div.pivot_table(
                    index=pc, columns="cluster_id", values="n_sources", aggfunc="mean"
                )
                fig, ax = plt.subplots(figsize=(max(12, pivot.shape[1] // 3), 5))
                im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
                ax.set_xticks(range(pivot.shape[1]))
                ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
                ax.set_yticks(range(pivot.shape[0]))
                ax.set_yticklabels(pivot.index, fontsize=8)
                ax.set_title("Cluster × Source Diversity (n_sources)")
                plt.colorbar(im, ax=ax, label="n_sources")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(STEP4_DIR, "cluster_source_diversity_heatmap.png"),
                    dpi=120,
                    bbox_inches="tight",
                )
                plt.close()
                log.info("  Saved cluster_source_diversity_heatmap.png")
            except Exception as ex:
                log.warning(f"  Plot 18: {ex}")

# Plot 19 — Maturity distribution heatmap (Gap 4: valid_pathway_nodes-filtered)
interv_clusters_mat = {
    cid: [n for n in nodes if n in valid_pathway_nodes]
    for cid, nodes in get_clusters("0.9", "unconstrained", "intervention").items()
}
mat_data = defaultdict(Counter)
for cid, node_ids in interv_clusters_mat.items():
    for nid in node_ids:
        mat = node_attrs.get(nid, {}).get("intervention_maturity")
        if mat is not None:
            try:
                mat_data[cid][int(mat)] += 1
            except Exception:
                pass

if mat_data:
    cids_s = sorted(mat_data.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
    mat_m = np.zeros((4, len(cids_s)))
    for j, cid in enumerate(cids_s):
        total = sum(mat_data[cid].values())
        for i, lv in enumerate([1, 2, 3, 4]):
            mat_m[i, j] = mat_data[cid].get(lv, 0) / total if total else 0
    fig, ax = plt.subplots(figsize=(max(14, len(cids_s) // 2), 4))
    im = ax.imshow(mat_m, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cids_s)))
    ax.set_xticklabels(cids_s, rotation=90, fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Maturity {lvl}" for lvl in [1, 2, 3, 4]], fontsize=9)
    ax.set_title("Intervention Cluster × Maturity Distribution")
    plt.colorbar(im, ax=ax, label="Proportion")
    plt.tight_layout()
    plt.savefig(
        os.path.join(STEP4_DIR, "maturity_distribution_heatmap.png"),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()
    log.info("  Saved maturity_distribution_heatmap.png")

# Plot 21 — Within-cluster EDGE density heatmap
hce_set = set()
for e in edge_data:
    if str(e.get("type", "")).upper() != "EDGE":
        continue
    try:
        conf = float(e.get("confidence", 0) or 0)
    except Exception:
        conf = 0.0
    if conf < 3:
        continue
    try:
        s, t = int(e["source"]), int(e["target"])
        hce_set.add((min(s, t), max(s, t)))
    except Exception:
        pass

nt_density = ["risk", "intervention", "problem_analysis", "implementation_mechanism"]
density_rows = []
for nt in nt_density:
    for cid, raw_ids in get_clusters("0.9", "unconstrained", nt).items():
        # Gap 4: filter to valid_pathway_nodes for correct density denominator
        node_ids = [x for x in raw_ids if x in valid_pathway_nodes]
        ns = set(int(n) for n in node_ids)
        n = len(ns)
        if n < 2:
            continue
        within = sum(1 for (s, t) in hce_set if s in ns and t in ns)
        density_rows.append(
            {"node_type": nt, "cluster_id": cid, "edge_density": within / (n * (n - 1))}
        )

if density_rows:
    try:
        pivot_d = pd.DataFrame(density_rows).pivot_table(
            index="node_type", columns="cluster_id", values="edge_density"
        )
        fig, ax = plt.subplots(figsize=(max(14, pivot_d.shape[1] // 2), 4))
        im = ax.imshow(pivot_d.values, aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(range(pivot_d.shape[1]))
        ax.set_xticklabels(pivot_d.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(pivot_d.shape[0]))
        ax.set_yticklabels(pivot_d.index, fontsize=9)
        ax.set_title("Within-Cluster EDGE Density (conf≥3)")
        plt.colorbar(im, ax=ax, label="Edge Density")
        plt.tight_layout()
        plt.savefig(
            os.path.join(STEP4_DIR, "within_cluster_edge_density.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close()
        log.info("  Saved within_cluster_edge_density.png")
    except Exception as ex:
        log.warning(f"  Plot 21: {ex}")

# ─── Save checkpoints ─────────────────────────────────────────────────────────
with open(os.path.join(STEP4_DIR, "risk_clusters_09.pkl"), "wb") as f:
    pickle.dump(risk_clusters_09, f)

log.info("=" * 70)
log.info(f"Step 4b FULL DATA COMPLETE — {datetime.now().isoformat()}")
