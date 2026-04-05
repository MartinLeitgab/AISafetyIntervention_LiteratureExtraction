"""
Phase 2 Step 4 — Connectivity Analysis and Subcluster Analysis (FULL DATA version)
- Connectivity: streams ALL VarB paths directly (not sampled file), predicts chain cluster via KMeans model
- Subclusters: no node cap (uses all nodes in each cluster)
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
from sklearn.cluster import AgglomerativeClustering

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "phase2_results")
STEP1_DIR = os.path.join(RESULTS_DIR, "step1_load_and_parse_umapwithoutlocalsatellites")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4_finalanalysis")
PATHS_DIR = os.path.join(ROOT, "phase1_rawpathsfiles")
LOG_DIR = os.path.join(ROOT, "logfiles", "phase4_logs")
OUT_CONN = os.path.join(STEP4_DIR, "step4_connectivity")
OUT_SUB = os.path.join(STEP4_DIR, "step4_subclusters")
OUT_TABLES = os.path.join(STEP4_DIR, "step4_cluster_tables")

for d in [OUT_CONN, OUT_SUB, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "phase4_connectivity_fulldata.log")
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
log.info("Phase 2 Step 4 — Connectivity Analysis (FULL DATA)")
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


# ─── Load data ────────────────────────────────────────────────────────────────
log.info("Loading PKL files …")
t0 = time.time()
with open(os.path.join(STEP1_DIR, "cluster_memberships.pkl"), "rb") as f:
    cm = pickle.load(f)
with open(os.path.join(STEP1_DIR, "graph_node_attributes.pkl"), "rb") as f:
    node_attrs = pickle.load(f)
with open(os.path.join(STEP1_DIR, "graph_edge_data.pkl"), "rb") as f:
    edge_data = pickle.load(f)
log.info(f"  Loaded in {time.time() - t0:.1f}s")

# Build SIM edge set for max_consec_SIM filtering (SIM>=0.9 only)
log.info("Building SIM edge set (SIM>=0.9 only) …")
t_sim = time.time()


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


sim_edge_set = set()
for e in edge_data:
    if str(e.get("type", "")).upper() == "SIMILARITY":
        score = e.get("similarity_score")
        if score is not None and cos_sim_from_score(score) >= 0.9:
            try:
                s, t = int(e["source"]), int(e["target"])
                sim_edge_set.add((min(s, t), max(s, t)))
            except (ValueError, TypeError):
                pass
log.info(
    f"  sim_edge_set (SIM>=0.9): {len(sim_edge_set)} pairs  ({time.time() - t_sim:.1f}s)"
)


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


# Build embedding cache
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

# ─── Load valid_pathway_nodes (Gap 4 — CRITICAL: was missing entirely) ────────
log.info("Loading valid_pathway_nodes …")
t_vp = time.time()
valid_pathway_nodes = set()
paths_file_vp = os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl")
with open(paths_file_vp, "r") as f:
    for line in f:
        obj = json.loads(line)
        for nid in obj["path"]:
            valid_pathway_nodes.add(int(nid))
log.info(
    f"  {len(valid_pathway_nodes)} valid-pathway nodes  ({time.time() - t_vp:.1f}s)"
)

# ─── Load Option A KMeans model ───────────────────────────────────────────────
kmeans_model_file = os.path.join(STEP4_DIR, "optionA_kmeans_model.pkl")
with open(kmeans_model_file, "rb") as f:
    kmeans = pickle.load(f)
log.info(f"  Loaded KMeans model: {kmeans.n_clusters} clusters")

# ─── Load risk_clusters_09 ────────────────────────────────────────────────────
with open(os.path.join(STEP4_DIR, "risk_clusters_09.pkl"), "rb") as f:
    risk_clusters_09_raw = pickle.load(f)
# Gap 3b: apply valid_pathway_nodes filter after loading (PKL may be unfiltered)
risk_clusters_09 = {
    cid: [n for n in nodes if n in valid_pathway_nodes]
    for cid, nodes in risk_clusters_09_raw.items()
}


# ─── Cluster helpers ──────────────────────────────────────────────────────────
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


# ─── Build node → cluster mappings ───────────────────────────────────────────
node_to_risk = {}
for cid, node_ids in risk_clusters_09.items():
    for nid in node_ids:
        node_to_risk[nid] = cid

interv_clusters_09_raw = get_clusters("0.9", "unconstrained", "intervention")
# Gap 4: apply valid_pathway_nodes filter (holistic — subsumes maturity≥3)
interv_clusters_09 = {
    cid: [n for n in nodes if n in valid_pathway_nodes]
    for cid, nodes in interv_clusters_09_raw.items()
}
node_to_interv = {}
for cid, node_ids in interv_clusters_09.items():
    for nid in node_ids:
        node_to_interv[nid] = cid

log.info(f"  risk node_to_cluster: {len(node_to_risk)} nodes")
log.info(f"  interv node_to_cluster: {len(node_to_interv)} nodes")

# ─── SECTION A: Connectivity — stream ALL VarB paths ─────────────────────────
log.info("=" * 50)
log.info(
    "SECTION A: Three-level hierarchy connectivity — ALL VarB paths (max_consec_SIM<=2)"
)

paths_file = os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl")

risk_to_chain = Counter()
chain_to_interv = Counter()
risk_to_interv = Counter()

PREDICT_BATCH = 5000
batch_embs = []
batch_meta = []  # list of (risk_cid, interv_cid)

n_paths_read = 0
n_varb = 0
n_chain_assigned = 0
n_no_cluster = 0

log.info("  Streaming full path file …")
t_conn = time.time()

with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        n_paths_read += 1

        mcs = max_consec_sim(path)
        if mcs > 2:
            continue
        n_varb += 1

        risk_node = path[0]
        interv_node = path[-1]
        risk_cid = node_to_risk.get(risk_node)
        interv_cid = node_to_interv.get(interv_node)

        if risk_cid is None or interv_cid is None:
            n_no_cluster += 1
            continue

        # Always count risk_to_interv
        risk_to_interv[(str(risk_cid), str(interv_cid))] += 1

        # Compute body mean embedding for chain cluster prediction
        body_ids = path[1:-1]
        embs_b = [emb_cache[nid] for nid in body_ids if nid in emb_cache]
        if not embs_b:
            continue
        mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
        batch_embs.append(mean_emb)
        batch_meta.append((risk_cid, interv_cid))

        if len(batch_embs) >= PREDICT_BATCH:
            labels = kmeans.predict(np.stack(batch_embs))
            for lab, (rc, ic) in zip(labels, batch_meta):
                chain_cid = str(int(lab))
                risk_to_chain[(str(rc), chain_cid)] += 1
                chain_to_interv[(chain_cid, str(ic))] += 1
                n_chain_assigned += 1
            batch_embs = []
            batch_meta = []

# Flush remaining batch
if batch_embs:
    labels = kmeans.predict(np.stack(batch_embs))
    for lab, (rc, ic) in zip(labels, batch_meta):
        chain_cid = str(int(lab))
        risk_to_chain[(str(rc), chain_cid)] += 1
        chain_to_interv[(chain_cid, str(ic))] += 1
        n_chain_assigned += 1

log.info(f"  Done in {time.time() - t_conn:.1f}s")
log.info(f"  Total paths read: {n_paths_read:,}")
log.info(f"  VarB paths (max_consec_SIM<=2): {n_varb:,}")
log.info(f"  Paths without cluster mapping: {n_no_cluster:,}")
log.info(f"  Chain-assigned paths: {n_chain_assigned:,}")
log.info(f"  risk_to_chain: {len(risk_to_chain)} edges")
log.info(f"  chain_to_interv: {len(chain_to_interv)} edges")
log.info(f"  risk_to_interv: {len(risk_to_interv)} edges")


# Save edge CSVs
def counter_to_df(counter):
    rows = [
        {"cluster_a": k[0], "cluster_b": k[1], "n_paths": v} for k, v in counter.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("n_paths", ascending=False)
        .reset_index(drop=True)
    )


r2c_df = counter_to_df(risk_to_chain)
r2c_df.to_csv(os.path.join(OUT_CONN, "risk_to_chain_edges.csv"), index=False)
log.info(f"  Saved risk_to_chain_edges.csv ({len(r2c_df)} rows)")

c2i_df = counter_to_df(chain_to_interv)
c2i_df.to_csv(os.path.join(OUT_CONN, "chain_to_intervention_edges.csv"), index=False)
log.info(f"  Saved chain_to_intervention_edges.csv ({len(c2i_df)} rows)")

r2i_df = counter_to_df(risk_to_interv)
r2i_df.to_csv(os.path.join(OUT_CONN, "risk_to_intervention_edges.csv"), index=False)
log.info(f"  Saved risk_to_intervention_edges.csv ({len(r2i_df)} rows)")

# ─── Gap Analysis ─────────────────────────────────────────────────────────────
log.info("Gap analysis …")

# Exclude clusters with zero valid_pathway_nodes-filtered members (Part 4 — empty in this config)
all_risk_clusters = set(str(c) for c in risk_clusters_09.keys() if risk_clusters_09[c])
all_interv_clusters = set(
    str(c) for c in interv_clusters_09.keys() if interv_clusters_09[c]
)
all_chain_clusters = set(str(i) for i in range(kmeans.n_clusters))

risk_with_chains = set(r2c_df["cluster_a"].astype(str)) if len(r2c_df) > 0 else set()
chain_with_risk = set(r2c_df["cluster_b"].astype(str)) if len(r2c_df) > 0 else set()
chain_with_interv = set(c2i_df["cluster_a"].astype(str)) if len(c2i_df) > 0 else set()
interv_with_chain = set(c2i_df["cluster_b"].astype(str)) if len(c2i_df) > 0 else set()
risk_with_interv_direct = (
    set(r2i_df["cluster_a"].astype(str)) if len(r2i_df) > 0 else set()
)
interv_with_risk_direct = (
    set(r2i_df["cluster_b"].astype(str)) if len(r2i_df) > 0 else set()
)

gap_rows = [
    {
        "gap_type": "risk_clusters_with_no_chain_connection",
        "count": len(all_risk_clusters - risk_with_chains),
        "examples": ",".join(sorted(all_risk_clusters - risk_with_chains)[:5]),
    },
    {
        "gap_type": "chain_clusters_with_no_risk_connection",
        "count": len(all_chain_clusters - chain_with_risk),
        "examples": ",".join(sorted(all_chain_clusters - chain_with_risk)[:5]),
    },
    {
        "gap_type": "chain_clusters_with_no_interv_connection",
        "count": len(all_chain_clusters - chain_with_interv),
        "examples": ",".join(sorted(all_chain_clusters - chain_with_interv)[:5]),
    },
    {
        "gap_type": "interv_clusters_with_no_chain_connection",
        "count": len(all_interv_clusters - interv_with_chain),
        "examples": ",".join(sorted(all_interv_clusters - interv_with_chain)[:5]),
    },
    {
        "gap_type": "risk_clusters_with_no_direct_interv_link",
        "count": len(all_risk_clusters - risk_with_interv_direct),
        "examples": ",".join(sorted(all_risk_clusters - risk_with_interv_direct)[:5]),
    },
    {
        "gap_type": "interv_clusters_with_no_direct_risk_link",
        "count": len(all_interv_clusters - interv_with_risk_direct),
        "examples": ",".join(sorted(all_interv_clusters - interv_with_risk_direct)[:5]),
    },
]
pd.DataFrame(gap_rows).to_csv(os.path.join(OUT_CONN, "gap_analysis.csv"), index=False)
log.info("  Saved gap_analysis.csv")
for g in gap_rows:
    log.info(f"    {g['gap_type']}: {g['count']}")

# ─── Three-layer Network Plot ──────────────────────────────────────────────────
log.info("Three-layer network plot …")

top20_risk = (
    r2i_df.groupby("cluster_a")["n_paths"].sum().nlargest(20).index.tolist()
    if len(r2i_df) > 0
    else []
)
top20_interv = (
    r2i_df.groupby("cluster_b")["n_paths"].sum().nlargest(20).index.tolist()
    if len(r2i_df) > 0
    else []
)

fig, ax = plt.subplots(figsize=(16, 10))
risk_y = {cid: i for i, cid in enumerate(top20_risk)}
interv_y = {cid: i for i, cid in enumerate(top20_interv)}

try:
    risk_names = pd.read_csv(os.path.join(OUT_TABLES, "risk_cluster_names.csv"))
    interv_names = pd.read_csv(
        os.path.join(OUT_TABLES, "intervention_cluster_names.csv")
    )
    rc_label = dict(
        zip(risk_names["cluster_id"].astype(str), risk_names["cluster_name"].str[:35])
    )
    ic_label = dict(
        zip(
            interv_names["cluster_id"].astype(str),
            interv_names["cluster_name"].str[:35],
        )
    )
except Exception:
    rc_label = {}
    ic_label = {}

for cid, y in risk_y.items():
    ax.scatter(0, y, s=200, c="steelblue", zorder=3)
    ax.text(
        -0.1, y, rc_label.get(str(cid), f"R{cid}"), ha="right", va="center", fontsize=6
    )

for cid, y in interv_y.items():
    ax.scatter(2, y, s=200, c="darkorange", zorder=3)
    ax.text(
        2.1, y, ic_label.get(str(cid), f"I{cid}"), ha="left", va="center", fontsize=6
    )

max_paths_val = r2i_df["n_paths"].max() if len(r2i_df) > 0 else 1
for _, row in r2i_df.head(100).iterrows():
    rc = str(row["cluster_a"])
    ic = str(row["cluster_b"])
    if rc in risk_y and ic in interv_y:
        ry, iy = risk_y[rc], interv_y[ic]
        lw = max(0.3, row["n_paths"] / max_paths_val * 6)
        ax.plot([0, 2], [ry, iy], "gray", alpha=0.35, linewidth=lw)

ax.set_xlim(-2.5, 4.5)
ax.set_title(
    f"Three-layer Connectivity: Risk → Intervention\n(top-100 edges from {n_varb:,} VarB paths — full data)"
)
ax.set_yticks([])
ax.legend(
    handles=[
        plt.scatter([], [], c="steelblue", s=100, label="Risk cluster"),
        plt.scatter([], [], c="darkorange", s=100, label="Intervention cluster"),
    ],
    loc="center",
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_CONN, "three_layer_network.png"), dpi=120, bbox_inches="tight"
)
plt.close()
log.info("  Saved three_layer_network.png")

# ─── SECTION B: Subcluster Analysis — ALL nodes, no cap ──────────────────────
log.info("=" * 50)
log.info("SECTION B: Subcluster Analysis (ALL nodes per cluster, no cap)")

sub_rows = []

for node_type_key in ["risk", "intervention"]:
    clusters = risk_clusters_09 if node_type_key == "risk" else interv_clusters_09
    clust_df = pd.read_csv(os.path.join(OUT_TABLES, f"{node_type_key}_clusters.csv"))
    csim_means = dict(
        zip(clust_df["cluster_id"].astype(str), clust_df["centroid_sim_mean"])
    )

    for cid, node_ids in clusters.items():
        cid_str = str(cid)
        n = len(node_ids)
        csim_mean = float(csim_means.get(cid_str, 1.0))

        cats = [
            node_attrs.get(nid, {}).get("concept_category", "") for nid in node_ids[:50]
        ]
        unique_cats = len(set(c for c in cats if c))

        needs_split = csim_mean < 0.3 or n > 100 or unique_cats > 2
        if not needs_split:
            continue

        log.info(
            f"  Subclustering {node_type_key} cluster {cid}: n={n}, csim={csim_mean:.3f}, cats={unique_cats}"
        )

        # Use ALL nodes — no cap
        embs = [emb_cache[nid] for nid in node_ids if nid in emb_cache]
        if len(embs) < 10:
            continue

        emb_matrix = np.stack(embs)
        n_sub = min(5, len(emb_matrix))
        if n_sub < 2:
            continue

        try:
            sub_labels = AgglomerativeClustering(
                n_clusters=n_sub, linkage="ward"
            ).fit_predict(emb_matrix)
            sub_cats = defaultdict(list)
            for i, lab in enumerate(sub_labels):
                if i < len(node_ids):
                    cat = node_attrs.get(node_ids[i], {}).get(
                        "concept_category", "unknown"
                    )
                    sub_cats[int(lab)].append(cat)
            sub_summary = {
                str(k): Counter(v).most_common(3) for k, v in sub_cats.items()
            }

            sub_rows.append(
                {
                    "node_type": node_type_key,
                    "cluster_id": cid_str,
                    "n_nodes": n,
                    "centroid_sim_mean": round(csim_mean, 4),
                    "n_unique_cats": unique_cats,
                    "split_reason": (
                        "low_csim"
                        if csim_mean < 0.3
                        else "large_size"
                        if n > 100
                        else "multi_cat"
                    ),
                    "n_subclusters": n_sub,
                    "sub_cluster_summary": str(sub_summary)[:300],
                }
            )
        except Exception as ex:
            log.warning(f"  Subcluster error for {cid}: {ex}")

log.info(f"  Found {len(sub_rows)} clusters needing subclustering")
pd.DataFrame(sub_rows).to_csv(
    os.path.join(OUT_SUB, "subcluster_summary.csv"), index=False
)
log.info("  Saved subcluster_summary.csv")

log.info("=" * 70)
log.info(f"Connectivity Analysis FULL DATA COMPLETE — {datetime.now().isoformat()}")
