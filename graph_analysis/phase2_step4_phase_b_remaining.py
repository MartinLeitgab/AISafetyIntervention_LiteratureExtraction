"""
Phase 2 Step 4 — Phase B Remaining (plan items 11-18)
Runs consim0 (k=10) and consim1 (k=40) chain body KMeans + connectivity + gap analysis.
Uses pre-filtered path files — NO edge_data loading needed.

Path file formats:
  paths_unconstrained_edge_only.jsonl : key='path'    (3,473 paths, mcs==0)
  representative_pathways_consim1.jsonl: key='node_id_sequence' (75,008 paths, mcs<=1)
  representative_pathways_consim2.jsonl: key='node_id_sequence' (432,776 paths, mcs<=2)
  paths_unconstrained_sim0.9.jsonl    : key='path'    (all 1,054,527 qualifying paths)

Outputs (step4_finalanalysis/):
  step4_cluster_tables/risk_clusters_consim{N}.csv
  step4_cluster_tables/interv_clusters_consim{N}.csv
  step4_cluster_tables/optionA_chainbody_clusters_consim{N}.csv
  step4_connectivity/risk_to_interv_edges_consim{N}.csv
  step4_connectivity/risk_to_chain_edges_consim{N}.csv
  step4_connectivity/chain_to_interv_edges_consim{N}.csv
  step4_connectivity/gap_analysis_consim{N}.csv
  step4_connectivity/cross_config_comparison.csv
  step4_config_selection.md (Part 2 criteria scoring)
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

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "phase2_results")
STEP1_DIR = os.path.join(RESULTS_DIR, "step1_load_and_parse_umapwithoutlocalsatellites")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4_finalanalysis")
PATHS_DIR = os.path.join(ROOT, "phase1_rawpathsfiles")
STEP4_PATHS = os.path.join(STEP4_DIR, "step4_paths")
OUT_TABLES = os.path.join(STEP4_DIR, "step4_cluster_tables")
OUT_CONN = os.path.join(STEP4_DIR, "step4_connectivity")
LOG_DIR = os.path.join(ROOT, "logfiles", "phase4_logs")

for d in [OUT_TABLES, OUT_CONN, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "phase4_phaseb_remaining.log")
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
log.info("Phase 2 Step 4 — Phase B Remaining (consim0 + consim1 + config selection)")
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


def get_path_ids(obj):
    """Handles both 'path' and 'node_id_sequence' key formats."""
    if "path" in obj:
        return [int(x) for x in obj["path"]]
    return [int(x) for x in obj["node_id_sequence"]]


def stream_path_file(filepath):
    """Yields list[int] path for each line."""
    with open(filepath, "r") as f:
        for line in f:
            yield get_path_ids(json.loads(line))


def counter_to_df(counter):
    rows = [
        {"cluster_a": k[0], "cluster_b": k[1], "n_paths": v} for k, v in counter.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("n_paths", ascending=False)
        .reset_index(drop=True)
    )


# ─── Load PKL ─────────────────────────────────────────────────────────────────
log.info("Loading PKL files (no edge_data — uses pre-filtered path files) …")
t0 = time.time()
with open(os.path.join(STEP1_DIR, "cluster_memberships.pkl"), "rb") as f:
    cm = pickle.load(f)
with open(os.path.join(STEP1_DIR, "graph_node_attributes.pkl"), "rb") as f:
    node_attrs = pickle.load(f)
log.info(
    f"  cm: {len(cm)} keys, node_attrs: {len(node_attrs)} nodes  ({time.time() - t0:.1f}s)"
)

# ─── Embedding cache ──────────────────────────────────────────────────────────
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


def get_qualifying_clusters(node_type, vpn):
    """Filter cluster members to config-specific valid_pathway_nodes."""
    raw = get_clusters("0.9", "unconstrained", node_type)
    return {cid: [n for n in nodes if n in vpn] for cid, nodes in raw.items()}


# ─── Build VPN sets ───────────────────────────────────────────────────────────
log.info("Building config-specific valid_pathway_nodes sets …")

consim0_vpn = set()
t = time.time()
for path in stream_path_file(
    os.path.join(PATHS_DIR, "paths_unconstrained_edge_only.jsonl")
):
    consim0_vpn.update(path)
log.info(f"  consim0_vpn: {len(consim0_vpn)} nodes  ({time.time() - t:.1f}s)")

consim1_vpn = set()
t = time.time()
for path in stream_path_file(
    os.path.join(STEP4_PATHS, "representative_pathways_consim1.jsonl")
):
    consim1_vpn.update(path)
log.info(f"  consim1_vpn: {len(consim1_vpn)} nodes  ({time.time() - t:.1f}s)")

consim2_vpn = set()
t = time.time()
for path in stream_path_file(
    os.path.join(STEP4_PATHS, "representative_pathways_consim2.jsonl")
):
    consim2_vpn.update(path)
log.info(f"  consim2_vpn: {len(consim2_vpn)} nodes  ({time.time() - t:.1f}s)")

# Unconstrained VPN: build from all 1,054,527 paths
unconstrained_vpn = set()
t = time.time()
with open(os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl"), "r") as f:
    for line in f:
        obj = json.loads(line)
        for nid in obj["path"]:
            unconstrained_vpn.add(int(nid))
log.info(
    f"  unconstrained_vpn: {len(unconstrained_vpn)} nodes  ({time.time() - t:.1f}s)"
)

# ─── VPN cluster coverage stats ───────────────────────────────────────────────
log.info("Computing VPN risk cluster coverage …")
raw_risk = get_clusters("0.9", "unconstrained", "risk")
for config_name, vpn in [
    ("consim0", consim0_vpn),
    ("consim1", consim1_vpn),
    ("consim2", consim2_vpn),
    ("unconstrained", unconstrained_vpn),
]:
    n_qual = sum(1 for nodes in raw_risk.values() for n in nodes if n in vpn)
    log.info(f"  {config_name}: {n_qual} qualifying risk PKL nodes")


# ─── Cluster table builder ────────────────────────────────────────────────────
def build_cluster_table_for_config(
    node_type, vpn, config_name, extra_maturity_filter=False
):
    """Build cluster table for a specific config VPN."""
    clusters = get_qualifying_clusters(node_type, vpn)
    if extra_maturity_filter:  # belt-and-suspenders for intervention
        clusters = {
            cid: [
                n
                for n in nodes
                if (node_attrs.get(n, {}).get("intervention_maturity") or 0) >= 3
            ]
            for cid, nodes in clusters.items()
        }

    rows = []
    for cid, node_ids in clusters.items():
        if not node_ids:
            continue
        embs = [emb_cache[nid] for nid in node_ids if nid in emb_cache]
        if not embs:
            continue
        centroid = np.stack(embs).mean(axis=0)
        csims = [
            cosine_sim(emb_cache[nid], centroid) for nid in node_ids if nid in emb_cache
        ]
        centroid_sim_mean = float(np.mean(csims)) if csims else 0.0

        ranked = sorted(
            [
                (cosine_sim(emb_cache[nid], centroid), nid)
                for nid in node_ids
                if nid in emb_cache
            ],
            reverse=True,
        )
        top5 = []
        for _, nid in ranked:
            is_dup = any(
                cosine_sim(emb_cache[nid], emb_cache[prev]) >= 0.95
                for prev in top5
                if prev in emb_cache and nid in emb_cache
            )
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

        # edge-only path fraction: fraction of config-qualifying nodes also in consim0_vpn
        n_consim0 = sum(1 for n in node_ids if n in consim0_vpn)
        edge_only_frac = round(n_consim0 / len(node_ids), 4) if node_ids else 0.0

        rows.append(
            {
                "cluster_id": cid,
                "n_nodes": len(node_ids),
                "n_sources": n_sources,
                "centroid_sim_mean": round(centroid_sim_mean, 4),
                "edge_only_path_fraction": edge_only_frac,
                "top5_names": top5_names,
                "top_node_name": str(top_node.get("name", ""))[:100],
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("n_nodes", ascending=False)
        .reset_index(drop=True)
    )
    fname = f"{node_type}_clusters_{config_name}.csv"
    df.to_csv(os.path.join(OUT_TABLES, fname), index=False)
    log.info(
        f"  Saved {fname} ({len(df)} clusters, {df['n_nodes'].sum()} qualifying nodes)"
    )
    return clusters


# ─── SECTION 1: consim0 Analysis (k=10 KMeans + connectivity + gap) ───────────
log.info("=" * 70)
log.info("SECTION 1: consim0 (edge-only) — k=10 KMeans + connectivity")

t_c0 = time.time()
CONSIM0_K = 10
CONSIM1_K = 40

risk_c0 = build_cluster_table_for_config("risk", consim0_vpn, "consim0")
interv_c0 = build_cluster_table_for_config(
    "intervention", consim0_vpn, "consim0", extra_maturity_filter=True
)

# node→cluster maps for consim0
node_to_risk_c0 = {nid: cid for cid, nodes in risk_c0.items() for nid in nodes}
node_to_interv_c0 = {nid: cid for cid, nodes in interv_c0.items() for nid in nodes}
log.info(
    f"  consim0 risk nodes: {len(node_to_risk_c0)}, interv nodes: {len(node_to_interv_c0)}"
)

# KMeans k=10 on consim0 body embeddings (2-pass streaming)
log.info(f"  Training MiniBatchKMeans k={CONSIM0_K} on consim0 paths …")
BATCH = 500
kmeans_c0 = MiniBatchKMeans(
    n_clusters=CONSIM0_K, random_state=42, batch_size=BATCH, n_init=10
)
batch_embs = []
n_paths_c0 = 0

for path in stream_path_file(
    os.path.join(PATHS_DIR, "paths_unconstrained_edge_only.jsonl")
):
    if len(path) < 3:
        continue
    body_ids = path[1:-1]
    embs_b = [emb_cache[nid] for nid in body_ids if nid in emb_cache]
    if not embs_b:
        continue
    mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
    batch_embs.append(mean_emb)
    n_paths_c0 += 1
    if len(batch_embs) >= BATCH:
        kmeans_c0.partial_fit(np.stack(batch_embs))
        batch_embs = []

if batch_embs:
    kmeans_c0.partial_fit(np.stack(batch_embs))
log.info(f"  Fitted on {n_paths_c0} consim0 paths")

# Pass 2: connectivity for consim0
risk_to_chain_c0 = Counter()
chain_to_interv_c0 = Counter()
risk_to_interv_c0 = Counter()
chain_data_c0 = defaultdict(lambda: {"n_paths": 0, "body_ids": set(), "urls": set()})
n_no_cluster_c0 = 0
batch_embs2, batch_meta2 = [], []

for path in stream_path_file(
    os.path.join(PATHS_DIR, "paths_unconstrained_edge_only.jsonl")
):
    if len(path) < 3:
        continue
    risk_node = path[0]
    interv_node = path[-1]
    rc = node_to_risk_c0.get(risk_node)
    ic = node_to_interv_c0.get(interv_node)
    if rc is None or ic is None:
        n_no_cluster_c0 += 1
        continue

    risk_to_interv_c0[(str(rc), str(ic))] += 1
    body_ids = path[1:-1]
    embs_b = [emb_cache[nid] for nid in body_ids if nid in emb_cache]
    if not embs_b:
        continue
    mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
    batch_embs2.append(mean_emb)
    batch_meta2.append((str(rc), str(ic), body_ids))

    if len(batch_embs2) >= BATCH:
        labels = kmeans_c0.predict(np.stack(batch_embs2))
        for lab, (r, i, bids) in zip(labels, batch_meta2):
            cc = str(int(lab))
            risk_to_chain_c0[(r, cc)] += 1
            chain_to_interv_c0[(cc, i)] += 1
            chain_data_c0[int(lab)]["n_paths"] += 1
            chain_data_c0[int(lab)]["body_ids"].update(bids)
        batch_embs2, batch_meta2 = [], []

if batch_embs2:
    labels = kmeans_c0.predict(np.stack(batch_embs2))
    for lab, (r, i, bids) in zip(labels, batch_meta2):
        cc = str(int(lab))
        risk_to_chain_c0[(r, cc)] += 1
        chain_to_interv_c0[(cc, i)] += 1
        chain_data_c0[int(lab)]["n_paths"] += 1
        chain_data_c0[int(lab)]["body_ids"].update(bids)

log.info(f"  consim0 no-cluster paths: {n_no_cluster_c0}")
log.info(
    f"  risk_to_chain: {len(risk_to_chain_c0)}, chain_to_interv: {len(chain_to_interv_c0)}"
)
log.info(f"  risk_to_interv: {len(risk_to_interv_c0)}")

# Save connectivity CSVs for consim0
for counter, name in [
    (risk_to_chain_c0, "risk_to_chain_edges_consim0.csv"),
    (chain_to_interv_c0, "chain_to_interv_edges_consim0.csv"),
    (risk_to_interv_c0, "risk_to_interv_edges_consim0.csv"),
]:
    df = counter_to_df(counter)
    df.to_csv(os.path.join(OUT_CONN, name), index=False)
    log.info(f"  Saved {name} ({len(df)} rows)")

# Chain cluster table for consim0
chain_rows_c0 = []
for cid, data in sorted(chain_data_c0.items()):
    body_ids_set = data["body_ids"]
    urls = set(str(node_attrs.get(n, {}).get("url", "")) for n in body_ids_set) - {
        "",
        "None",
        "nan",
    }
    embs_c = [emb_cache[nid] for nid in body_ids_set if nid in emb_cache]
    centroid_c = np.stack(embs_c).mean(axis=0) if embs_c else None
    chain_rows_c0.append(
        {
            "cluster_id": cid,
            "n_paths": data["n_paths"],
            "n_unique_body_nodes": len(body_ids_set),
            "n_sources": len(urls),
        }
    )
pd.DataFrame(chain_rows_c0).sort_values("n_paths", ascending=False).to_csv(
    os.path.join(OUT_TABLES, "optionA_chainbody_clusters_consim0.csv"), index=False
)
log.info("  Saved optionA_chainbody_clusters_consim0.csv")

# Gap analysis for consim0
all_risk_c0 = set(cid for cid, nodes in risk_c0.items() if nodes)
all_interv_c0 = set(cid for cid, nodes in interv_c0.items() if nodes)
all_chain_c0 = set(str(i) for i in range(CONSIM0_K))

r2c_c0 = counter_to_df(risk_to_chain_c0)
c2i_c0 = counter_to_df(chain_to_interv_c0)
r2i_c0 = counter_to_df(risk_to_interv_c0)

gap_c0 = [
    (
        "risk_clusters_with_no_chain_connection",
        len(all_risk_c0 - set(r2c_c0["cluster_a"].astype(str) if len(r2c_c0) else [])),
        ",".join(
            sorted(
                all_risk_c0
                - set(r2c_c0["cluster_a"].astype(str) if len(r2c_c0) else [])
            )[:5]
        ),
    ),
    (
        "chain_clusters_with_no_risk_connection",
        len(all_chain_c0 - set(r2c_c0["cluster_b"].astype(str) if len(r2c_c0) else [])),
        ",".join(
            sorted(
                all_chain_c0
                - set(r2c_c0["cluster_b"].astype(str) if len(r2c_c0) else [])
            )[:5]
        ),
    ),
    (
        "chain_clusters_with_no_interv_connection",
        len(all_chain_c0 - set(c2i_c0["cluster_a"].astype(str) if len(c2i_c0) else [])),
        ",".join(
            sorted(
                all_chain_c0
                - set(c2i_c0["cluster_a"].astype(str) if len(c2i_c0) else [])
            )[:5]
        ),
    ),
    (
        "interv_clusters_with_no_chain_connection",
        len(
            all_interv_c0 - set(c2i_c0["cluster_b"].astype(str) if len(c2i_c0) else [])
        ),
        ",".join(
            sorted(
                all_interv_c0
                - set(c2i_c0["cluster_b"].astype(str) if len(c2i_c0) else [])
            )[:5]
        ),
    ),
    (
        "risk_clusters_with_no_direct_interv_link",
        len(all_risk_c0 - set(r2i_c0["cluster_a"].astype(str) if len(r2i_c0) else [])),
        ",".join(
            sorted(
                all_risk_c0
                - set(r2i_c0["cluster_a"].astype(str) if len(r2i_c0) else [])
            )[:5]
        ),
    ),
    (
        "interv_clusters_with_no_direct_risk_link",
        len(
            all_interv_c0 - set(r2i_c0["cluster_b"].astype(str) if len(r2i_c0) else [])
        ),
        ",".join(
            sorted(
                all_interv_c0
                - set(r2i_c0["cluster_b"].astype(str) if len(r2i_c0) else [])
            )[:5]
        ),
    ),
]
gap_df_c0 = pd.DataFrame(gap_c0, columns=["gap_type", "count", "examples"])
gap_df_c0.to_csv(os.path.join(OUT_CONN, "gap_analysis_consim0.csv"), index=False)
log.info("  Saved gap_analysis_consim0.csv")
for _, row in gap_df_c0.iterrows():
    log.info(f"    {row['gap_type']}: {row['count']}  examples: {row['examples'][:60]}")

log.info(f"SECTION 1 done in {time.time() - t_c0:.1f}s")


# ─── SECTION 2: consim1 Analysis (k=40 KMeans + connectivity + gap) ───────────
log.info("=" * 70)
log.info("SECTION 2: consim1 (max_consec_SIM<=1) — k=40 KMeans + connectivity")

t_c1 = time.time()
risk_c1 = build_cluster_table_for_config("risk", consim1_vpn, "consim1")
interv_c1 = build_cluster_table_for_config(
    "intervention", consim1_vpn, "consim1", extra_maturity_filter=True
)

node_to_risk_c1 = {nid: cid for cid, nodes in risk_c1.items() for nid in nodes}
node_to_interv_c1 = {nid: cid for cid, nodes in interv_c1.items() for nid in nodes}
log.info(
    f"  consim1 risk nodes: {len(node_to_risk_c1)}, interv nodes: {len(node_to_interv_c1)}"
)

CONSIM1_PATH_FILE = os.path.join(STEP4_PATHS, "representative_pathways_consim1.jsonl")

# KMeans k=40 on consim1 body embeddings — 2-pass streaming
log.info(f"  Training MiniBatchKMeans k={CONSIM1_K} on consim1 paths …")
BATCH1 = 5000
kmeans_c1 = MiniBatchKMeans(
    n_clusters=CONSIM1_K, random_state=42, batch_size=BATCH1, n_init=10
)
batch_embs = []
n_paths_c1 = 0

for path in stream_path_file(CONSIM1_PATH_FILE):
    if len(path) < 3:
        continue
    body_ids = path[1:-1]
    embs_b = [emb_cache[nid] for nid in body_ids if nid in emb_cache]
    if not embs_b:
        continue
    mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
    batch_embs.append(mean_emb)
    n_paths_c1 += 1
    if len(batch_embs) >= BATCH1:
        kmeans_c1.partial_fit(np.stack(batch_embs))
        batch_embs = []

if batch_embs:
    kmeans_c1.partial_fit(np.stack(batch_embs))
log.info(f"  Fitted on {n_paths_c1} consim1 paths")

# Pass 2: connectivity for consim1
risk_to_chain_c1 = Counter()
chain_to_interv_c1 = Counter()
risk_to_interv_c1 = Counter()
chain_data_c1 = defaultdict(lambda: {"n_paths": 0, "body_ids": set(), "urls": set()})
n_no_cluster_c1 = 0
batch_embs2, batch_meta2 = [], []

for path in stream_path_file(CONSIM1_PATH_FILE):
    if len(path) < 3:
        continue
    risk_node = path[0]
    interv_node = path[-1]
    rc = node_to_risk_c1.get(risk_node)
    ic = node_to_interv_c1.get(interv_node)
    if rc is None or ic is None:
        n_no_cluster_c1 += 1
        continue

    risk_to_interv_c1[(str(rc), str(ic))] += 1
    body_ids = path[1:-1]
    embs_b = [emb_cache[nid] for nid in body_ids if nid in emb_cache]
    if not embs_b:
        continue
    mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
    batch_embs2.append(mean_emb)
    batch_meta2.append((str(rc), str(ic), body_ids))

    if len(batch_embs2) >= BATCH1:
        labels = kmeans_c1.predict(np.stack(batch_embs2))
        for lab, (r, i, bids) in zip(labels, batch_meta2):
            cc = str(int(lab))
            risk_to_chain_c1[(r, cc)] += 1
            chain_to_interv_c1[(cc, i)] += 1
            chain_data_c1[int(lab)]["n_paths"] += 1
            chain_data_c1[int(lab)]["body_ids"].update(bids)
        batch_embs2, batch_meta2 = [], []

if batch_embs2:
    labels = kmeans_c1.predict(np.stack(batch_embs2))
    for lab, (r, i, bids) in zip(labels, batch_meta2):
        cc = str(int(lab))
        risk_to_chain_c1[(r, cc)] += 1
        chain_to_interv_c1[(cc, i)] += 1
        chain_data_c1[int(lab)]["n_paths"] += 1
        chain_data_c1[int(lab)]["body_ids"].update(bids)

log.info(f"  consim1 no-cluster paths: {n_no_cluster_c1}")
log.info(
    f"  risk_to_chain: {len(risk_to_chain_c1)}, chain_to_interv: {len(chain_to_interv_c1)}"
)
log.info(f"  risk_to_interv: {len(risk_to_interv_c1)}")

# Save connectivity CSVs for consim1
for counter, name in [
    (risk_to_chain_c1, "risk_to_chain_edges_consim1.csv"),
    (chain_to_interv_c1, "chain_to_interv_edges_consim1.csv"),
    (risk_to_interv_c1, "risk_to_interv_edges_consim1.csv"),
]:
    df = counter_to_df(counter)
    df.to_csv(os.path.join(OUT_CONN, name), index=False)
    log.info(f"  Saved {name} ({len(df)} rows)")

# Chain cluster table for consim1
chain_rows_c1 = []
for cid, data in sorted(chain_data_c1.items()):
    body_ids_set = data["body_ids"]
    urls = set(str(node_attrs.get(n, {}).get("url", "")) for n in body_ids_set) - {
        "",
        "None",
        "nan",
    }
    chain_rows_c1.append(
        {
            "cluster_id": cid,
            "n_paths": data["n_paths"],
            "n_unique_body_nodes": len(body_ids_set),
            "n_sources": len(urls),
        }
    )
pd.DataFrame(chain_rows_c1).sort_values("n_paths", ascending=False).to_csv(
    os.path.join(OUT_TABLES, "optionA_chainbody_clusters_consim1.csv"), index=False
)
log.info("  Saved optionA_chainbody_clusters_consim1.csv")

# Gap analysis for consim1
all_risk_c1 = set(cid for cid, nodes in risk_c1.items() if nodes)
all_interv_c1 = set(cid for cid, nodes in interv_c1.items() if nodes)
all_chain_c1 = set(str(i) for i in range(CONSIM1_K))

r2c_c1 = counter_to_df(risk_to_chain_c1)
c2i_c1 = counter_to_df(chain_to_interv_c1)
r2i_c1 = counter_to_df(risk_to_interv_c1)

gap_c1 = [
    (
        "risk_clusters_with_no_chain_connection",
        len(all_risk_c1 - set(r2c_c1["cluster_a"].astype(str) if len(r2c_c1) else [])),
        ",".join(
            sorted(
                all_risk_c1
                - set(r2c_c1["cluster_a"].astype(str) if len(r2c_c1) else [])
            )[:5]
        ),
    ),
    (
        "chain_clusters_with_no_risk_connection",
        len(all_chain_c1 - set(r2c_c1["cluster_b"].astype(str) if len(r2c_c1) else [])),
        ",".join(
            sorted(
                all_chain_c1
                - set(r2c_c1["cluster_b"].astype(str) if len(r2c_c1) else [])
            )[:5]
        ),
    ),
    (
        "chain_clusters_with_no_interv_connection",
        len(all_chain_c1 - set(c2i_c1["cluster_a"].astype(str) if len(c2i_c1) else [])),
        ",".join(
            sorted(
                all_chain_c1
                - set(c2i_c1["cluster_a"].astype(str) if len(c2i_c1) else [])
            )[:5]
        ),
    ),
    (
        "interv_clusters_with_no_chain_connection",
        len(
            all_interv_c1 - set(c2i_c1["cluster_b"].astype(str) if len(c2i_c1) else [])
        ),
        ",".join(
            sorted(
                all_interv_c1
                - set(c2i_c1["cluster_b"].astype(str) if len(c2i_c1) else [])
            )[:5]
        ),
    ),
    (
        "risk_clusters_with_no_direct_interv_link",
        len(all_risk_c1 - set(r2i_c1["cluster_a"].astype(str) if len(r2i_c1) else [])),
        ",".join(
            sorted(
                all_risk_c1
                - set(r2i_c1["cluster_a"].astype(str) if len(r2i_c1) else [])
            )[:5]
        ),
    ),
    (
        "interv_clusters_with_no_direct_risk_link",
        len(
            all_interv_c1 - set(r2i_c1["cluster_b"].astype(str) if len(r2i_c1) else [])
        ),
        ",".join(
            sorted(
                all_interv_c1
                - set(r2i_c1["cluster_b"].astype(str) if len(r2i_c1) else [])
            )[:5]
        ),
    ),
]
gap_df_c1 = pd.DataFrame(gap_c1, columns=["gap_type", "count", "examples"])
gap_df_c1.to_csv(os.path.join(OUT_CONN, "gap_analysis_consim1.csv"), index=False)
log.info("  Saved gap_analysis_consim1.csv")
for _, row in gap_df_c1.iterrows():
    log.info(f"    {row['gap_type']}: {row['count']}  examples: {row['examples'][:60]}")

log.info(f"SECTION 2 done in {time.time() - t_c1:.1f}s")


# ─── SECTION 3: Cross-config comparison ───────────────────────────────────────
log.info("=" * 70)
log.info("SECTION 3: Cross-config comparison")

# Load existing consim2 gap analysis
gap_c2_file = os.path.join(OUT_CONN, "gap_analysis.csv")
gap_df_c2 = pd.read_csv(gap_c2_file)
# Use consim2 VPN to get qualifying cluster membership counts
risk_c2 = {
    cid: [n for n in nodes if n in consim2_vpn]
    for cid, nodes in get_clusters("0.9", "unconstrained", "risk").items()
}
interv_c2 = {
    cid: [
        n
        for n in nodes
        if n in consim2_vpn
        and (node_attrs.get(n, {}).get("intervention_maturity") or 0) >= 3
    ]
    for cid, nodes in get_clusters("0.9", "unconstrained", "intervention").items()
}

# Summary rows for cross-config comparison
configs_summary = []
for config_name, vpn, risk_clusters, interv_clusters, n_paths_total, gap_df in [
    ("consim0 (edge-only)", consim0_vpn, risk_c0, interv_c0, n_paths_c0, gap_df_c0),
    ("consim1 (≤1 SIM hop)", consim1_vpn, risk_c1, interv_c1, n_paths_c1, gap_df_c1),
    ("consim2 (≤2 SIM hops)", consim2_vpn, risk_c2, interv_c2, 432776, gap_df_c2),
]:
    n_risk_qual = sum(len(v) for v in risk_clusters.values())
    n_interv_qual = sum(len(v) for v in interv_clusters.values())
    n_risk_nonempty = sum(1 for v in risk_clusters.values() if v)
    n_interv_nonempty = sum(1 for v in interv_clusters.values() if v)
    total_gaps = gap_df["count"].sum()
    configs_summary.append(
        {
            "config": config_name,
            "n_paths": n_paths_total,
            "n_qualifying_risk_nodes": n_risk_qual,
            "n_qualifying_interv_nodes": n_interv_qual,
            "n_nonempty_risk_clusters": n_risk_nonempty,
            "n_nonempty_interv_clusters": n_interv_nonempty,
            "total_gap_count": total_gaps,
        }
    )

summary_df = pd.DataFrame(configs_summary)
summary_df.to_csv(os.path.join(OUT_CONN, "cross_config_comparison.csv"), index=False)
log.info("  Saved cross_config_comparison.csv")
log.info(f"\n{summary_df.to_string(index=False)}")

# Gap breakdown table
gap_types = gap_df_c0["gap_type"].tolist()
gap_comparison_rows = []
for gap_type in gap_types:
    c0_val = gap_df_c0.loc[gap_df_c0["gap_type"] == gap_type, "count"].values[0]
    c1_val = gap_df_c1.loc[gap_df_c1["gap_type"] == gap_type, "count"].values[0]
    c2_val_rows = gap_df_c2.loc[gap_df_c2["gap_type"] == gap_type, "count"]
    c2_val = c2_val_rows.values[0] if len(c2_val_rows) > 0 else 0
    gap_comparison_rows.append(
        {
            "gap_type": gap_type,
            "consim0": c0_val,
            "consim1": c1_val,
            "consim2": c2_val,
        }
    )
gap_comparison_df = pd.DataFrame(gap_comparison_rows)
gap_comparison_df.to_csv(
    os.path.join(OUT_CONN, "gap_comparison_by_config.csv"), index=False
)
log.info("  Saved gap_comparison_by_config.csv")
log.info(f"\n{gap_comparison_df.to_string(index=False)}")

# Edge-only path fraction per cluster (consim1 and consim2)
edgefrac_rows = []
for config_name, clusters in [("consim1", risk_c1), ("consim2", risk_c2)]:
    for cid, node_ids in clusters.items():
        if not node_ids:
            continue
        n_consim0 = sum(1 for n in node_ids if n in consim0_vpn)
        frac = round(n_consim0 / len(node_ids), 4)
        edgefrac_rows.append(
            {
                "config": config_name,
                "node_type": "risk",
                "cluster_id": cid,
                "n_qualifying": len(node_ids),
                "n_consim0_anchored": n_consim0,
                "edge_only_frac": frac,
            }
        )

for config_name, clusters in [("consim1", interv_c1), ("consim2", interv_c2)]:
    for cid, node_ids in clusters.items():
        if not node_ids:
            continue
        n_consim0 = sum(1 for n in node_ids if n in consim0_vpn)
        frac = round(n_consim0 / len(node_ids), 4)
        edgefrac_rows.append(
            {
                "config": config_name,
                "node_type": "intervention",
                "cluster_id": cid,
                "n_qualifying": len(node_ids),
                "n_consim0_anchored": n_consim0,
                "edge_only_frac": frac,
            }
        )

edgefrac_df = pd.DataFrame(edgefrac_rows)
edgefrac_df.to_csv(os.path.join(OUT_CONN, "edge_only_frac_by_config.csv"), index=False)
log.info("  Saved edge_only_frac_by_config.csv")

# Summary stats per config
for config_name in ["consim1", "consim2"]:
    for ntype in ["risk", "intervention"]:
        sub = edgefrac_df[
            (edgefrac_df["config"] == config_name) & (edgefrac_df["node_type"] == ntype)
        ]
        if not sub.empty:
            mean_frac = sub["edge_only_frac"].mean()
            log.info(f"  {config_name} {ntype} mean edge_only_frac: {mean_frac:.3f}")

log.info("=" * 70)
log.info("Phase B Remaining complete")
log.info(f"Total runtime: {time.time() - t0:.1f}s")
log.info(f"End: {datetime.now().isoformat()}")
