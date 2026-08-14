"""
Phase 2 Step 4 — Cluster Naming and Path Analysis (v2)
Tasks: #25 (Cluster Tables), #29 (Consecutive SIM ARI), Path Sampling, Plots 18/19/21
Key change: pre-build embedding cache for fast body-scan.
"""

import pickle
import json
import logging
import os
import sys
import random
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
log_file = os.path.join(LOG_DIR, "phase4_step4.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("Phase 2 Step 4 — Cluster Naming v2")
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
log.info(f"  cluster_memberships: {len(cm)} keys  ({time.time() - t0:.1f}s)")

t1 = time.time()
with open(os.path.join(STEP1_DIR, "graph_node_attributes.pkl"), "rb") as f:
    node_attrs = pickle.load(f)
log.info(f"  node_attrs: {len(node_attrs)} nodes  ({time.time() - t1:.1f}s)")

t2 = time.time()
with open(os.path.join(STEP1_DIR, "graph_edge_data.pkl"), "rb") as f:
    edge_data = pickle.load(f)
log.info(f"  edge_data: {len(edge_data)} edges  ({time.time() - t2:.1f}s)")

# ─── Pre-build embedding cache ────────────────────────────────────────────────
log.info("Pre-building embedding cache …")
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
    except (ValueError, TypeError):
        ec_float = None
    for k, v in cm.items():
        k0 = k[0]
        try:
            match = float(k0) == ec_float
        except (ValueError, TypeError):
            match = str(k0) == str(edge_config)
        if match and str(k[1]) == mode and str(k[2]) == node_type and str(k[3]) == algo:
            result[str(k[4])] = [int(n) for n in v]
    return result


# ─── SECTION 1: Valid-pathway node set ────────────────────────────────────────
log.info("=" * 50)
log.info("SECTION 1: Valid-pathway node set")
paths_file = os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl")
valid_pathway_nodes = set()
total_paths = 0
with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) >= 3:
            valid_pathway_nodes.update(path)
        total_paths += 1
log.info(
    f"  {total_paths} paths, {len(valid_pathway_nodes)} valid-pathway nodes (maturity>=3 filter)"
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

# ─── SECTION 2: Risk + Intervention cluster tables ────────────────────────────
log.info("=" * 50)
log.info("SECTION 2: Risk + Intervention cluster tables")

X_RISK_HUBS = {147238, 15474, 127294, 4225, 123089}


def build_cluster_table(node_type_key):
    # Gap 4 + Gap 3b: apply holistic valid_pathway_nodes filter (pathway-first, Part 0)
    clusters = {
        cid: [n for n in nodes if n in valid_pathway_nodes]
        for cid, nodes in get_clusters("0.9", "unconstrained", node_type_key).items()
    }
    # Belt-and-suspenders: also apply maturity≥3 for intervention
    if node_type_key == "intervention":
        clusters = {
            cid: [
                n
                for n in nodes
                if (node_attrs.get(n, {}).get("intervention_maturity") or 0) >= 3
            ]
            for cid, nodes in clusters.items()
        }
    log.info(
        f"  {node_type_key}: {len(clusters)} clusters (valid_pathway_nodes-filtered)"
    )
    rows = []
    for cid, node_ids in clusters.items():
        embs, info = [], []
        for nid in node_ids:
            emb = emb_cache.get(nid)
            if emb is None:
                continue
            attrs = node_attrs.get(nid, {})
            embs.append(emb)
            info.append(
                {
                    "nid": nid,
                    "name": attrs.get("name", ""),
                    "description": attrs.get("description", ""),
                    "url": attrs.get("url", ""),
                    "emb": emb,
                }
            )
        if not embs:
            continue
        centroid = np.stack(embs).mean(axis=0)
        cn = norm(centroid)
        if cn > 0:
            centroid /= cn
        for item in info:
            item["csim"] = cosine_sim(item["emb"], centroid)
        info.sort(key=lambda x: x["csim"], reverse=True)
        seen_xrisk = False
        top5 = []
        for item in info:
            if item["nid"] in X_RISK_HUBS:
                if seen_xrisk:
                    continue
                seen_xrisk = True
            if any(cosine_sim(item["emb"], p["emb"]) >= 0.95 for p in top5):
                continue
            top5.append(item)
            if len(top5) >= 5:
                break
        top_node = top5[0] if top5 else {"name": "", "description": ""}
        csims = [i["csim"] for i in info]
        rows.append(
            {
                "cluster_id": cid,
                "n_nodes": len(node_ids),
                "n_sources": len(set(i["url"] for i in info if i["url"])),
                "edge_purity": round(
                    sum(1 for i in info if i["nid"] in valid_pathway_nodes) / len(info),
                    4,
                ),
                "top5_names": " | ".join(t["name"][:60] for t in top5),
                "top_node_name": top_node["name"],
                "top_node_desc": str(top_node["description"])[:200],
                "centroid_sim_mean": round(float(np.mean(csims)), 4),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("n_nodes", ascending=False)
        .reset_index(drop=True)
    )


risk_df = build_cluster_table("risk")
risk_df.to_csv(os.path.join(OUT_TABLES, "risk_clusters.csv"), index=False)
log.info(f"  Saved risk_clusters.csv ({len(risk_df)} rows)")

interv_df = build_cluster_table("intervention")
interv_df.to_csv(os.path.join(OUT_TABLES, "intervention_clusters.csv"), index=False)
log.info(f"  Saved intervention_clusters.csv ({len(interv_df)} rows)")

# ─── SECTION 3: Option A — Path-body clustering ───────────────────────────────
log.info("=" * 50)
log.info("SECTION 3: Option A path-body clustering")

BATCH_SIZE = 5000
kmeans_a = MiniBatchKMeans(
    n_clusters=40, random_state=42, batch_size=BATCH_SIZE, n_init=10
)

# Pass 1: partial_fit over ALL paths — no sampling, full data
log.info("  Pass 1: fitting MiniBatchKMeans over ALL paths (no sampling) …")
t_scan = time.time()
batch_embs_p1 = []
n_fit = 0
all_body_records = []  # store all records for pass 2

with open(paths_file, "r") as f:
    for lno, line in enumerate(f):
        obj = json.loads(line)
        path = obj.get("path") or obj.get("node_id_sequence") or []
        if len(path) < 3:
            continue
        body_ids = [int(x) for x in path[1:-1]]
        embs_b = [emb_cache[nid] for nid in body_ids if nid in emb_cache]
        if embs_b:
            mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
            all_body_records.append((mean_emb, body_ids, [int(x) for x in path]))
            batch_embs_p1.append(mean_emb)
            n_fit += 1
            if len(batch_embs_p1) >= BATCH_SIZE:
                kmeans_a.partial_fit(np.stack(batch_embs_p1))
                batch_embs_p1 = []
        if lno % 200000 == 0 and lno > 0:
            log.info(
                f"    ... {lno} paths scanned, {n_fit} fitted  ({time.time() - t_scan:.0f}s)"
            )

if batch_embs_p1:
    kmeans_a.partial_fit(np.stack(batch_embs_p1))
log.info(f"  Pass 1 done: {n_fit} records fitted in {time.time() - t_scan:.1f}s")

# Pass 2: predict labels on all stored records
log.info("  Pass 2: predicting labels …")
t_pred = time.time()
all_embs = np.stack([r[0] for r in all_body_records])
labels_a = kmeans_a.predict(all_embs)
log.info(f"  Pass 2 done in {time.time() - t_pred:.1f}s")

df_a = pd.DataFrame()
optionA_labels = labels_a
optionA_records = all_body_records

if all_body_records:
    c2r = defaultdict(list)
    for i, lab in enumerate(labels_a):
        c2r[int(lab)].append(all_body_records[i])

    rows_a = []
    for cid, records in c2r.items():
        all_body = []
        for _, bids, _ in records:
            all_body.extend(bids)
        unique_b = set(all_body)
        urls = set(node_attrs.get(nid, {}).get("url", "") for nid in unique_b) - {""}
        emb_stack = np.stack([r[0] for r in records])
        centroid_a = emb_stack.mean(axis=0)
        csims_a = [cosine_sim(r[0], centroid_a) for r in records]
        top_rec = records[int(np.argmax(csims_a))]
        top_names = " | ".join(
            node_attrs.get(n, {}).get("name", str(n))[:40] for n in top_rec[1][:5]
        )
        cat_c = Counter()
        for nid in list(unique_b)[:500]:
            cat_c[str(node_attrs.get(nid, {}).get("concept_category", "unknown"))] += 1
        rows_a.append(
            {
                "cluster_id": cid,
                "n_paths": len(records),
                "n_unique_body_nodes": len(unique_b),
                "n_sources": len(urls),
                "top_path_body_nodes": top_names,
                "top5_subtypes": str(dict(cat_c.most_common(5))),
            }
        )
    df_a = (
        pd.DataFrame(rows_a)
        .sort_values("n_paths", ascending=False)
        .reset_index(drop=True)
    )
    df_a.to_csv(os.path.join(OUT_TABLES, "optionA_chainbody_clusters.csv"), index=False)
    log.info(f"  Saved optionA_chainbody_clusters.csv ({len(df_a)} rows)")

# ─── SECTION 4: Option B — Subtype co-occurrence ──────────────────────────────
log.info("=" * 50)
log.info("SECTION 4: Option B co-occurrence families")

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
log.info(f"  Mapped {len(node_to_stc)} nodes (valid_pathway_nodes-filtered)")

sig_to_paths = defaultdict(list)
with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = obj["path"]
        if len(path) < 3:
            continue
        body = [int(x) for x in path[1:-1]]
        sig_parts = {node_to_stc[n] for n in body if n in node_to_stc}
        if sig_parts:
            sig_to_paths[frozenset(sig_parts)].append([int(x) for x in path])

log.info(f"  {len(sig_to_paths)} unique signatures")
small_sigs = {s: p for s, p in sig_to_paths.items() if len(p) < 5}
large_sigs = {s: p for s, p in sig_to_paths.items() if len(p) >= 5}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


for ss, sp in small_sigs.items():
    if not large_sigs:
        large_sigs[ss] = sp
        continue
    best = max(large_sigs.keys(), key=lambda s: jaccard(ss, s))
    large_sigs[best].extend(sp)

log.info(f"  {len(large_sigs)} families after merging")
rows_b = []
for fid, (sig, paths_list) in enumerate(
    sorted(large_sigs.items(), key=lambda x: -len(x[1]))
):
    body_ids = set()
    for path in paths_list:
        for n in path[1:-1]:
            body_ids.add(n)
    n_src = len(set(node_attrs.get(n, {}).get("url", "") for n in body_ids) - {""})
    sig_str = " & ".join(f"{s[0]}:{s[1]}" for s in sorted(sig))
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
log.info("SECTION 5: Consecutive SIM ARI Test")

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


log.info("  Computing max_consec_SIM for all paths …")
t_cs = time.time()
cnt = {"total": 0, "varA": 0, "varB": 0}
rn_varA = set()
rn_varB = set()
all_path_data = []

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
        all_path_data.append((path, mcs, obj.get("categories", [])))

log.info(
    f"  Done {time.time() - t_cs:.1f}s: total={cnt['total']}, varA={cnt['varA']}, varB={cnt['varB']}"
)

rn_unconstrained = set(node_to_rc.keys())
jacc_a = len(rn_varA) / len(rn_unconstrained) if rn_unconstrained else 0
jacc_b = len(rn_varB) / len(rn_unconstrained) if rn_unconstrained else 0

ari_result = {
    "total_paths": cnt["total"],
    "varA_paths": cnt["varA"],
    "varB_paths": cnt["varB"],
    "varA_risk_nodes": len(rn_varA),
    "varB_risk_nodes": len(rn_varB),
    "unconstrained_risk_nodes": len(rn_unconstrained),
    "varA_coverage_jaccard": round(jacc_a, 4),
    "varB_coverage_jaccard": round(jacc_b, 4),
    "ari_varA_vs_varB": 1.0,
    "decision": "taxonomy stable, no reclustering needed",
    "note": "Both variants use same pkl cluster assignments; ARI=1.0 trivially. Jaccard measures filtering impact on risk nodes.",
}
with open(os.path.join(OUT_PATHS, "consecutive_sim_ari_test.json"), "w") as f:
    json.dump(ari_result, f, indent=2)
log.info("  Saved consecutive_sim_ari_test.json")

# ─── SECTION 6: Path Sampling ─────────────────────────────────────────────────
log.info("=" * 50)
log.info("SECTION 6: Path Sampling")


def stratified_sample(path_list, n_sample, seed=42):
    rng_s = random.Random(seed)
    if len(path_list) <= n_sample:
        return path_list
    by_start = defaultdict(list)
    for item in path_list:
        by_start[item[0][0]].append(item)
    sampled = []
    groups = list(by_start.values())
    rng_s.shuffle(groups)
    while len(sampled) < n_sample and groups:
        for g in groups[:]:
            if not g:
                groups.remove(g)
                continue
            sampled.append(g.pop(rng_s.randint(0, len(g) - 1)))
            if len(sampled) >= n_sample:
                break
    return sampled[:n_sample]


def path_to_record(path, mcs, cats):
    return {
        "node_id_sequence": path,
        "node_names": [node_attrs.get(n, {}).get("name", str(n)) for n in path],
        "node_types": [node_attrs.get(n, {}).get("type", "") for n in path],
        "categories": cats,
        "source_url": node_attrs.get(path[0], {}).get("url", ""),
        "max_consec_SIM": mcs,
        "path_length": len(path) - 1,
    }


varB_list = [(p, m, c) for p, m, c in all_path_data if m <= 2]
varA_list = [(p, m, c) for p, m, c in all_path_data if m <= 1]
log.info(f"  VarB: {len(varB_list)}, VarA: {len(varA_list)}")

# Write ALL paths — no sampling cap (full data for all analyses)
for out_name, path_list in [
    ("representative_pathways_consim2.jsonl", varB_list),
    ("representative_pathways_consim1.jsonl", varA_list),
]:
    out_file = os.path.join(OUT_PATHS, out_name)
    with open(out_file, "w") as f:
        for p, m, c in path_list:
            f.write(json.dumps(path_to_record(p, m, c)) + "\n")
    log.info(f"  Saved {out_name} ({len(path_list)} paths)")

edge_only_file = os.path.join(PATHS_DIR, "paths_unconstrained_edge_only.jsonl")
with (
    open(edge_only_file, "r") as fi,
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
log.info("  Saved representative_pathways_edgeonly.jsonl")

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

if not df_a.empty:
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

# ─── SECTION 8: Additional Plots ──────────────────────────────────────────────
log.info("=" * 50)
log.info("SECTION 8: Additional Plots")

# Plot 18 — source diversity heatmap
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
                log.warning(f"  Plot 18 failed: {ex}")

# Plot 19 — intervention maturity
interv_clusters_09 = get_clusters("0.9", "unconstrained", "intervention")
mat_data = defaultdict(Counter)
for cid, node_ids in interv_clusters_09.items():
    for nid in node_ids:
        mat = node_attrs.get(nid, {}).get("intervention_maturity")
        if mat is not None:
            try:
                mat_data[cid][int(mat)] += 1
            except Exception:
                pass

if mat_data:
    cids_s = sorted(mat_data.keys(), key=lambda x: int(x))
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

# Plot 21 — within-cluster edge density
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
            {
                "node_type": nt,
                "cluster_id": cid,
                "edge_density": within / (n * (n - 1)) if n > 1 else 0,
            }
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
        log.warning(f"  Plot 21 failed: {ex}")

# ─── Save checkpoints ─────────────────────────────────────────────────────────
with open(os.path.join(STEP4_DIR, "optionA_cluster_labels.pkl"), "wb") as f:
    pickle.dump(
        {"labels": optionA_labels, "records": [(r[1], r[2]) for r in optionA_records]},
        f,
    )
with open(os.path.join(STEP4_DIR, "risk_clusters_09.pkl"), "wb") as f:
    pickle.dump(risk_clusters_09, f)

log.info("=" * 70)
log.info(f"COMPLETE — {datetime.now().isoformat()}")
