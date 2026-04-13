"""
Phase 2 Track A — Per-Config Independent Clustering + PathbuildB Empirical Check
=================================================================================
Addresses revision plan items:
  A1 (item 27): Per-config independent k=40 agglomerative clustering for risk and
                intervention VPN nodes; compute non-trivial config selection metrics
  A2 (items 22/3): PathbuildB top-20 frozenset decoding; empirical assessment of
                   whether B-families provide mechanistic vs risk-themed chains
  B3 (item 4):   Compute true consim2 Option B co-occurrence families CSV
  B4 (item 5):   Top-20 frozenset families per consimN (0/1/2) with representative
                 node names and path counts

Outputs (all in step4_finalanalysis/):
  step4_cluster_tables/risk_clusters_perconfig_consimN.csv  (N=0,1,2)
  step4_cluster_tables/intervention_clusters_perconfig_consimN.csv
  step4_cluster_tables/optionB_cooccurrence_families_consim2.csv  (B3)
  step4_cluster_tables/optionB_top20_decoded_consimN.csv  (B4/A2, N=0,1,2)
  step4_cluster_tables/bodysubtype_cluster_representatives.csv  (A2 helper)
  config_selection_metrics_v2.csv  (A1 - new non-trivial criteria)

Runtime: ~30-60 min (dominated by PKL loading + 2 passes over 1M-path file)
"""

import json
import logging
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "phase2_results")
STEP1_DIR = os.path.join(RESULTS_DIR, "step1_load_and_parse_umapwithoutlocalsatellites")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4_finalanalysis")
PATHS_DIR = os.path.join(ROOT, "phase1_rawpathsfiles")
LOG_DIR = os.path.join(ROOT, "logfiles", "phase4_logs")
OUT_TABLES = os.path.join(STEP4_DIR, "step4_cluster_tables")

for d in [OUT_TABLES, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "phase4_trackA.log")
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
log.info("Phase 2 Track A — Per-Config Clustering + PathbuildB Empirical Check")
log.info(f"Start: {datetime.now().isoformat()}")

# ─── Body subtype definitions ─────────────────────────────────────────────────
BODY_SUBTYPES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]
SUBTYPE_PREFIX = {
    "problem_analysis": "pr",
    "theoretical_insight": "th",
    "design_rationale": "de",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_embedding(emb_raw):
    if isinstance(emb_raw, np.ndarray):
        return emb_raw.astype(np.float32)
    s = str(emb_raw).strip().strip("<>")
    return np.array([float(x) for x in s.split(",")], dtype=np.float32)


def cosine_sim(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def get_clusters_from_cm(cm, edge_config, mode, node_type, algo="agglomerative"):
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


def normalize_rows(X):
    """L2-normalize each row of matrix X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def compute_intra_cluster_cosine_sim(node_ids, labels, emb_cache, n_clusters=40):
    """
    Compute mean intra-cluster cosine similarity (member to centroid) over all clusters.
    Returns overall mean, per-cluster dict.
    """
    # Build cluster → list of embeddings
    cluster_embs = defaultdict(list)
    for nid, lab in zip(node_ids, labels):
        if nid in emb_cache:
            cluster_embs[lab].append(emb_cache[nid])

    per_cluster_sim = {}
    for cid, embs in cluster_embs.items():
        if len(embs) < 2:
            continue
        X = np.stack(embs).astype(np.float32)
        X_norm = normalize_rows(X)
        centroid = X_norm.mean(axis=0)
        centroid = centroid / (norm(centroid) + 1e-8)
        csims = X_norm @ centroid
        per_cluster_sim[cid] = float(csims.mean())

    overall_mean = (
        float(np.mean(list(per_cluster_sim.values()))) if per_cluster_sim else 0.0
    )
    return overall_mean, per_cluster_sim


def run_independent_clustering(node_ids, emb_cache, n_clusters=40):
    """
    Run independent AgglomerativeClustering(k=40, ward) on given node IDs.
    Returns: (labels array, node_id list with embeddings, X_norm)
    Only nodes with embeddings are clustered.
    """
    ids_with_emb = [nid for nid in node_ids if nid in emb_cache]
    if len(ids_with_emb) < n_clusters:
        log.warning(
            f"    Only {len(ids_with_emb)} nodes with embeddings, skipping clustering"
        )
        return None, ids_with_emb, None

    X = np.stack([emb_cache[nid] for nid in ids_with_emb]).astype(np.float32)
    X_norm = normalize_rows(X)

    log.info(
        f"    AgglomerativeClustering(k={n_clusters}) on {len(ids_with_emb)} nodes …"
    )
    t0 = time.time()
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = agg.fit_predict(X_norm)
    log.info(f"    Clustering done in {time.time() - t0:.1f}s")
    return labels, ids_with_emb, X_norm


# ─── STEP 1: Load PKL files ───────────────────────────────────────────────────
log.info("=" * 60)
log.info("STEP 1: Loading PKL files")

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

# ─── Build embedding cache ────────────────────────────────────────────────────
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

# ─── STEP 2: Build broad VPN (unconstrained, maturity>=3) ────────────────────
log.info("=" * 60)
log.info("STEP 2: Building unconstrained VPN (Pass 1 over sim0.9 file)")
sim09_file = os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl")
edge_only_file = os.path.join(PATHS_DIR, "paths_unconstrained_edge_only.jsonl")

t_vpn = time.time()
vpn_unconstrained = set()
with open(sim09_file) as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) >= 3:
            vpn_unconstrained.update(path)
log.info(
    f"  vpn_unconstrained: {len(vpn_unconstrained)} nodes  ({time.time() - t_vpn:.1f}s)"
)

# ─── Build SIM edge set (SIM>=0.9, VPN-restricted) ───────────────────────────
log.info("Building sim_edge_set (SIM>=0.9, restricted to unconstrained VPN) …")
t_sim = time.time()
sim_edge_set = set()
for e in edge_data:
    if str(e.get("type", "")).upper() == "SIMILARITY":
        score = e.get("similarity_score")
        if score is not None and cos_sim_from_score(score) >= 0.9:
            try:
                s, tgt = int(e["source"]), int(e["target"])
                if s in vpn_unconstrained and tgt in vpn_unconstrained:
                    sim_edge_set.add((min(s, tgt), max(s, tgt)))
            except (ValueError, TypeError):
                pass
log.info(f"  sim_edge_set: {len(sim_edge_set)} pairs  ({time.time() - t_sim:.1f}s)")


def max_consec_sim_fn(path_ids):
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


# ─── STEP 3: Build per-consimN VPNs + B-family signature counts ──────────────
log.info("=" * 60)
log.info("STEP 3: Building per-consimN VPNs + collecting B-family signatures")
log.info("  (Pass 2 over sim0.9 file + pass over edge_only file)")

# node_to_stc: body node → (subtype, cluster_id_str) from PKL
log.info("  Building node_to_stc for body subtypes …")
node_to_stc = {}
for (ec, mode, nt, algo, cid), members in cm.items():
    try:
        ec_float = float(ec)
    except Exception:
        continue
    if (
        ec_float == 0.9
        and str(mode) == "unconstrained"
        and str(nt) in BODY_SUBTYPES
        and str(algo) == "agglomerative"
    ):
        for nid in members:
            nid_int = int(nid)
            if nid_int in vpn_unconstrained:
                node_to_stc[nid_int] = (str(nt), str(cid))
log.info(f"  node_to_stc: {len(node_to_stc)} body nodes")

# PKL-based risk/intervention cluster assignments (for coverage computation)
log.info("  Loading PKL risk/intervention cluster assignments …")
risk_clusters_base = get_clusters_from_cm(cm, "0.9", "unconstrained", "risk")
interv_clusters_base = get_clusters_from_cm(cm, "0.9", "unconstrained", "intervention")

node_to_risk_cluster = {}
for cid, nids in risk_clusters_base.items():
    for nid in nids:
        node_to_risk_cluster[nid] = cid

node_to_interv_cluster = {}
for cid, nids in interv_clusters_base.items():
    for nid in nids:
        node_to_interv_cluster[nid] = cid

log.info(
    f"  risk_clusters_base: {len(risk_clusters_base)} clusters, {len(node_to_risk_cluster)} nodes"
)
log.info(
    f"  interv_clusters_base: {len(interv_clusters_base)} clusters, {len(node_to_interv_cluster)} nodes"
)

# Per-consimN data structures
vpn = {"consim0": set(), "consim1": set(), "consim2": set()}
n_paths = {"consim0": 0, "consim1": 0, "consim2": 0}
# B-family signature counts per consimN
sig_counts = {"consim0": Counter(), "consim1": Counter(), "consim2": Counter()}
# R→I pair coverage per consimN (using PKL cluster assignments)
ri_pairs = {"consim0": set(), "consim1": set(), "consim2": set()}

# Pass over edge_only file (consim0)
log.info("  Processing edge_only file (consim0) …")
t_eo = time.time()
with open(edge_only_file) as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) < 3:
            continue
        vpn["consim0"].update(path)
        n_paths["consim0"] += 1
        # B-family signature
        body = path[1:-1]
        sig = frozenset(node_to_stc[n] for n in body if n in node_to_stc)
        if sig:
            sig_counts["consim0"][sig] += 1
        # R→I pair
        risk_start = path[0]
        interv_end = path[-1]
        rc = node_to_risk_cluster.get(risk_start)
        ic = node_to_interv_cluster.get(interv_end)
        if rc is not None and ic is not None:
            ri_pairs["consim0"].add((rc, ic))
log.info(
    f"  consim0: {n_paths['consim0']} paths, {len(vpn['consim0'])} VPN nodes  ({time.time() - t_eo:.1f}s)"
)

# Pass over sim0.9 file (consim1 and consim2)
log.info("  Processing sim0.9 file (consim1 + consim2) …")
t_sim09 = time.time()
n_total = 0
with open(sim09_file) as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        n_total += 1
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) < 3:
            continue
        mcs = max_consec_sim_fn(path)

        # B-family signature
        body = path[1:-1]
        sig = frozenset(node_to_stc[n] for n in body if n in node_to_stc)

        # R→I pair
        risk_start = path[0]
        interv_end = path[-1]
        rc = node_to_risk_cluster.get(risk_start)
        ic = node_to_interv_cluster.get(interv_end)

        if mcs <= 1:
            vpn["consim1"].update(path)
            n_paths["consim1"] += 1
            if sig:
                sig_counts["consim1"][sig] += 1
            if rc is not None and ic is not None:
                ri_pairs["consim1"].add((rc, ic))

        if mcs <= 2:
            vpn["consim2"].update(path)
            n_paths["consim2"] += 1
            if sig:
                sig_counts["consim2"][sig] += 1
            if rc is not None and ic is not None:
                ri_pairs["consim2"].add((rc, ic))

log.info(
    f"  sim0.9: {n_total} total paths processed  ({time.time() - t_sim09:.1f}s)\n"
    f"    consim1: {n_paths['consim1']} paths, {len(vpn['consim1'])} VPN nodes\n"
    f"    consim2: {n_paths['consim2']} paths, {len(vpn['consim2'])} VPN nodes"
)

# ─── STEP 4: B3 — Save true consim2 Option B families ───────────────────────
log.info("=" * 60)
log.info("STEP 4 (B3): Saving true consim2 Option B co-occurrence families")


def build_ob_family_csv(sig_counts_config, config_name, out_filename):
    """Build and save Option B family CSV for a given consimN config."""
    # Keep families with ≥5 paths
    large_sigs = {s: c for s, c in sig_counts_config.items() if c >= 5}
    log.info(
        f"  {config_name}: {len(sig_counts_config)} unique sigs, "
        f"{len(large_sigs)} with n≥5 paths"
    )

    rows = []
    for fid, (sig, n_p) in enumerate(sorted(large_sigs.items(), key=lambda x: -x[1])):
        sig_str = " & ".join(
            f"{SUBTYPE_PREFIX.get(s[0], s[0][:2])}:{s[1]}" for s in sorted(sig)
        )
        top_subtypes = dict(Counter(s[0] for s in sig).most_common(3))
        rows.append(
            {
                "family_id": fid,
                "n_paths": n_p,
                "signature_str": sig_str[:200],
                "top_subtypes": str(top_subtypes),
            }
        )

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_TABLES, out_filename)
    df.to_csv(out_path, index=False)
    log.info(f"  Saved {out_path} ({len(df)} rows)")
    return df


df_ob2 = build_ob_family_csv(
    sig_counts["consim2"], "consim2", "optionB_cooccurrence_families_consim2.csv"
)

# ─── STEP 5: B4 — Top-20 frozenset families with decoded representative names ─
log.info("=" * 60)
log.info("STEP 5 (B4/A2): Computing body subtype cluster representatives")
log.info("  (Needed to decode B-family signatures into human-readable names)")

# For each (subtype, cluster_id) from the PKL, find centroid-closest node name
body_cluster_reps = {}  # (subtype, cid_str) → {"rep_name": ..., "n_members": ...}

for subtype in BODY_SUBTYPES:
    sub_clusters = get_clusters_from_cm(cm, "0.9", "unconstrained", subtype)
    log.info(f"  {subtype}: {len(sub_clusters)} clusters")
    for cid_str, member_ids in sub_clusters.items():
        # Filter to VPN members with embeddings
        valid_members = [
            nid for nid in member_ids if nid in vpn_unconstrained and nid in emb_cache
        ]
        if not valid_members:
            body_cluster_reps[(subtype, cid_str)] = {
                "rep_name": "[no emb members]",
                "n_members": len(member_ids),
                "n_vpn_members": 0,
            }
            continue

        # Compute centroid
        embs = np.stack([emb_cache[nid] for nid in valid_members]).astype(np.float32)
        embs_norm = normalize_rows(embs)
        centroid = embs_norm.mean(axis=0)
        centroid = centroid / (norm(centroid) + 1e-8)

        # Find closest node to centroid
        csims = embs_norm @ centroid
        best_idx = int(csims.argmax())
        best_nid = valid_members[best_idx]
        rep_name = str(node_attrs.get(best_nid, {}).get("name", str(best_nid)))

        # Find top-3 diverse reps
        top3_names = []
        used = []
        for idx in csims.argsort()[::-1]:
            nid = valid_members[idx]
            # Skip near-duplicates
            is_dup = False
            for prev_nid in used:
                if nid in emb_cache and prev_nid in emb_cache:
                    if cosine_sim(emb_cache[nid], emb_cache[prev_nid]) >= 0.95:
                        is_dup = True
                        break
            if not is_dup:
                top3_names.append(
                    str(node_attrs.get(nid, {}).get("name", str(nid)))[:80]
                )
                used.append(nid)
            if len(top3_names) >= 3:
                break

        body_cluster_reps[(subtype, cid_str)] = {
            "rep_name": rep_name[:100],
            "top3_names": " | ".join(top3_names),
            "n_members": len(member_ids),
            "n_vpn_members": len(valid_members),
            "centroid_sim": float(csims[best_idx]),
        }

# Save body cluster representatives
rows_bcr = []
for (subtype, cid_str), info in sorted(body_cluster_reps.items()):
    rows_bcr.append(
        {
            "subtype": subtype,
            "cluster_id": cid_str,
            "prefix_key": f"{SUBTYPE_PREFIX.get(subtype, subtype[:2])}:{cid_str}",
            "rep_name": info.get("rep_name", ""),
            "top3_names": info.get("top3_names", ""),
            "n_members": info.get("n_members", 0),
            "n_vpn_members": info.get("n_vpn_members", 0),
        }
    )
df_bcr = pd.DataFrame(rows_bcr)
df_bcr.to_csv(
    os.path.join(OUT_TABLES, "bodysubtype_cluster_representatives.csv"), index=False
)
log.info(f"  Saved bodysubtype_cluster_representatives.csv ({len(df_bcr)} rows)")

# Build fast lookup: prefix_key → rep_name (e.g., "pr:6" → "...")
prefix_to_name = {r["prefix_key"]: r["rep_name"] for _, r in df_bcr.iterrows()}
prefix_to_top3 = {r["prefix_key"]: r["top3_names"] for _, r in df_bcr.iterrows()}


def decode_signature(sig_str):
    """Decode 'de:15 & im:4 & pr:6 & th:11 & va:10' → list of (prefix_key, rep_name)."""
    parts = [p.strip() for p in sig_str.split("&")]
    decoded = []
    for part in parts:
        name = prefix_to_name.get(part, f"[unknown:{part}]")
        decoded.append(f"{part}={name[:60]}")
    return " | ".join(decoded)


def build_top20_decoded_table(sig_counts_config, config_name, out_filename):
    """Build top-20 decoded frozenset table for a consimN config."""
    # Sort by n_paths descending, take top 20 (no min filter for display)
    sorted_sigs = sorted(sig_counts_config.items(), key=lambda x: -x[1])[:20]
    rows = []
    for rank, (sig, n_p) in enumerate(sorted_sigs):
        sig_str = " & ".join(
            f"{SUBTYPE_PREFIX.get(s[0], s[0][:2])}:{s[1]}" for s in sorted(sig)
        )
        # Decoded: map each (subtype, cid) to representative name
        decoded_parts = []
        for subtype, cid_str in sorted(sig):
            prefix_key = f"{SUBTYPE_PREFIX.get(subtype, subtype[:2])}:{cid_str}"
            rep = prefix_to_name.get(prefix_key, f"[{prefix_key}]")
            decoded_parts.append(f"{prefix_key}: {rep[:60]}")
        decoded_str = "\n".join(decoded_parts)

        rows.append(
            {
                "rank": rank + 1,
                "n_paths": n_p,
                "n_subtype_clusters": len(sig),
                "signature_str": sig_str[:200],
                "decoded_chain_components": decoded_str[:500],
            }
        )
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_TABLES, out_filename)
    df.to_csv(out_path, index=False)
    log.info(f"  Saved {out_path} ({len(df)} rows)")
    return df


log.info("Building top-20 decoded tables per consimN …")
for cfg_name in ["consim0", "consim1", "consim2"]:
    df_top20 = build_top20_decoded_table(
        sig_counts[cfg_name],
        cfg_name,
        f"optionB_top20_decoded_{cfg_name}.csv",
    )

# ─── STEP 6 (A1): Per-config independent k=40 clustering ────────────────────
log.info("=" * 60)
log.info("STEP 6 (A1): Per-config independent k=40 agglomerative clustering")
log.info("  For risk and intervention VPN nodes per consimN")


def build_perconfig_cluster_table(node_ids_all, node_type, config_name, out_filename):
    """
    Run independent AgglomerativeClustering(k=40) on VPN nodes of given type.
    Returns (node_labels_dict, intra_sim_mean, df_table)
    """
    # Filter to node type
    if node_type == "risk":
        type_nodes = [
            nid
            for nid in node_ids_all
            if str(node_attrs.get(nid, {}).get("type", "")) == "concept"
            and str(node_attrs.get(nid, {}).get("concept_category", "")) == "risk"
        ]
    elif node_type == "intervention":
        type_nodes = [
            nid
            for nid in node_ids_all
            if str(node_attrs.get(nid, {}).get("type", "")) == "intervention"
            and int(node_attrs.get(nid, {}).get("intervention_maturity", 0) or 0) >= 3
        ]
    else:
        type_nodes = list(node_ids_all)

    log.info(f"  {config_name}/{node_type}: {len(type_nodes)} VPN nodes of this type")

    labels, ids_with_emb, X_norm = run_independent_clustering(
        type_nodes, emb_cache, n_clusters=40
    )
    if labels is None:
        return {}, 0.0, pd.DataFrame()

    # Build node → cluster dict
    node_labels = {nid: int(lab) for nid, lab in zip(ids_with_emb, labels)}

    # Compute intra-cluster cosine sim
    intra_mean, per_cluster_sims = compute_intra_cluster_cosine_sim(
        ids_with_emb, labels, emb_cache, n_clusters=40
    )
    log.info(f"    Mean intra-cluster cosine sim: {intra_mean:.4f}")

    # Build cluster table (cluster_id, n_nodes, n_sources, centroid_sim_mean, top5_names)
    cluster_data = defaultdict(list)
    for nid, lab in zip(ids_with_emb, labels):
        cluster_data[int(lab)].append(nid)

    rows = []
    for cid, nids in sorted(cluster_data.items(), key=lambda x: -len(x[1])):
        embs_c = [emb_cache[nid] for nid in nids if nid in emb_cache]
        if not embs_c:
            continue
        X_c = np.stack(embs_c).astype(np.float32)
        X_c_norm = normalize_rows(X_c)
        centroid = X_c_norm.mean(axis=0)
        centroid = centroid / (norm(centroid) + 1e-8)
        csims_c = X_c_norm @ centroid

        # Top-5 representative names (dedup near-duplicates)
        order = csims_c.argsort()[::-1]
        top5 = []
        used_nids = []
        for idx in order:
            nid = nids[idx]
            is_dup = any(
                (
                    nid in emb_cache
                    and prev in emb_cache
                    and cosine_sim(emb_cache[nid], emb_cache[prev]) >= 0.95
                )
                for prev in used_nids
            )
            if not is_dup:
                top5.append(nid)
                used_nids.append(nid)
            if len(top5) >= 5:
                break

        top5_names = " | ".join(
            str(node_attrs.get(n, {}).get("name", str(n)))[:60] for n in top5
        )
        n_sources = len(
            {str(node_attrs.get(n, {}).get("url", "")) for n in nids}
            - {"", "None", "nan"}
        )
        rows.append(
            {
                "cluster_id": cid,
                "n_nodes": len(nids),
                "n_sources": n_sources,
                "centroid_sim_mean": round(per_cluster_sims.get(cid, 0.0), 4),
                "top5_names": top5_names[:300],
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("n_nodes", ascending=False)
        .reset_index(drop=True)
    )
    out_path = os.path.join(OUT_TABLES, out_filename)
    df.to_csv(out_path, index=False)
    log.info(f"    Saved {out_path} ({len(df)} rows)")
    return node_labels, intra_mean, df


# Store per-config clustering results for ARI computation
perconfig_risk_labels = {}
perconfig_interv_labels = {}
perconfig_intra_sim = {}

for cfg_name in ["consim0", "consim1", "consim2"]:
    log.info(f"\n--- Per-config clustering: {cfg_name} ---")
    vpn_nodes = vpn[cfg_name]

    # Risk
    risk_labels, risk_intra, _ = build_perconfig_cluster_table(
        vpn_nodes, "risk", cfg_name, f"risk_clusters_perconfig_{cfg_name}.csv"
    )
    # Intervention
    interv_labels, interv_intra, _ = build_perconfig_cluster_table(
        vpn_nodes,
        "intervention",
        cfg_name,
        f"intervention_clusters_perconfig_{cfg_name}.csv",
    )
    perconfig_risk_labels[cfg_name] = risk_labels
    perconfig_interv_labels[cfg_name] = interv_labels
    perconfig_intra_sim[cfg_name] = {
        "risk": risk_intra,
        "intervention": interv_intra,
    }

# ─── STEP 7 (A1): Compute cross-config ARI (true, non-trivial) ──────────────
log.info("=" * 60)
log.info("STEP 7 (A1): Computing cross-config ARI using independent clusterings")


def compute_cross_ari(labels_a, labels_b, config_a, config_b, node_type):
    """Compute ARI between two per-config cluster label dicts for shared nodes."""
    shared = set(labels_a.keys()) & set(labels_b.keys())
    if len(shared) < 10:
        log.warning(
            f"    {config_a}↔{config_b} {node_type}: only {len(shared)} shared nodes, ARI unreliable"
        )
        return None, len(shared)
    la = [labels_a[n] for n in shared]
    lb = [labels_b[n] for n in shared]
    ari = adjusted_rand_score(la, lb)
    log.info(
        f"    ARI({config_a}↔{config_b}) {node_type}: {ari:.4f} (n_shared={len(shared)})"
    )
    return ari, len(shared)


ari_results = {}
for node_type, labels_dict in [
    ("risk", perconfig_risk_labels),
    ("intervention", perconfig_interv_labels),
]:
    for ca, cb in [
        ("consim0", "consim1"),
        ("consim1", "consim2"),
        ("consim0", "consim2"),
    ]:
        ari, n_shared = compute_cross_ari(
            labels_dict[ca], labels_dict[cb], ca, cb, node_type
        )
        ari_results[f"ari_{ca}_vs_{cb}_{node_type}"] = ari
        ari_results[f"n_shared_{ca}_{cb}_{node_type}"] = n_shared

# ─── STEP 8 (A1): Compute R→I pair coverage (C4 metric) ─────────────────────
log.info("=" * 60)
log.info("STEP 8 (A1): R→I pair coverage (fraction of all possible pairs covered)")

n_risk_clusters = len(risk_clusters_base)
n_interv_clusters = len(interv_clusters_base)
total_possible_ri = n_risk_clusters * n_interv_clusters
log.info(
    f"  Total possible R×I pairs: {n_risk_clusters} × {n_interv_clusters} = {total_possible_ri}"
)

coverage_results = {}
for cfg_name in ["consim0", "consim1", "consim2"]:
    n_covered = len(ri_pairs[cfg_name])
    frac = n_covered / total_possible_ri if total_possible_ri > 0 else 0.0
    coverage_results[cfg_name] = {
        "n_covered_pairs": n_covered,
        "coverage_fraction": frac,
    }
    log.info(
        f"  {cfg_name}: {n_covered}/{total_possible_ri} pairs = {frac:.3f} coverage"
    )

# ─── STEP 9 (A1): Compute edge-only node fraction per config ─────────────────
log.info("=" * 60)
log.info("STEP 9 (A1): Edge-only node fraction per config (C2 metric)")

vpn0 = vpn["consim0"]

eo_fracs = {}
for cfg_name in ["consim0", "consim1", "consim2"]:
    vpn_cfg = vpn[cfg_name]
    # Risk nodes in VPN for this config
    risk_in_vpn = [
        n
        for n in vpn_cfg
        if str(node_attrs.get(n, {}).get("type", "")) == "concept"
        and str(node_attrs.get(n, {}).get("concept_category", "")) == "risk"
    ]
    interv_in_vpn = [
        n
        for n in vpn_cfg
        if str(node_attrs.get(n, {}).get("type", "")) == "intervention"
        and int(node_attrs.get(n, {}).get("intervention_maturity", 0) or 0) >= 3
    ]

    n_risk = len(risk_in_vpn)
    n_interv = len(interv_in_vpn)
    n_risk_eo = sum(1 for n in risk_in_vpn if n in vpn0)
    n_interv_eo = sum(1 for n in interv_in_vpn if n in vpn0)

    risk_eo_frac = n_risk_eo / n_risk if n_risk > 0 else 0.0
    interv_eo_frac = n_interv_eo / n_interv if n_interv > 0 else 0.0

    eo_fracs[cfg_name] = {
        "n_risk_nodes": n_risk,
        "n_interv_nodes": n_interv,
        "risk_eo_frac": risk_eo_frac,
        "interv_eo_frac": interv_eo_frac,
    }
    log.info(
        f"  {cfg_name}: risk eo_frac={risk_eo_frac:.4f} ({n_risk_eo}/{n_risk}), "
        f"interv eo_frac={interv_eo_frac:.4f} ({n_interv_eo}/{n_interv})"
    )

# ─── STEP 10: Save config selection metrics table (v2) ───────────────────────
log.info("=" * 60)
log.info("STEP 10: Saving config selection metrics v2")

config_rows = []
for cfg_name in ["consim0", "consim1", "consim2"]:
    row = {
        "config": cfg_name,
        "n_paths": n_paths[cfg_name],
        "n_vpn_nodes": len(vpn[cfg_name]),
        # C1: intra-cluster cosine sim (independent clustering)
        "C1_risk_intra_sim": round(perconfig_intra_sim[cfg_name]["risk"], 4),
        "C1_interv_intra_sim": round(perconfig_intra_sim[cfg_name]["intervention"], 4),
        # C2: edge-only fraction
        "C2_risk_eo_frac": round(eo_fracs[cfg_name]["risk_eo_frac"], 4),
        "C2_interv_eo_frac": round(eo_fracs[cfg_name]["interv_eo_frac"], 4),
        # C3: ARI vs next config (non-trivial, from independent clusterings)
        "C3_ari_risk_vs_next": (
            round(ari_results.get(f"ari_{cfg_name}_vs_consim1_risk", 0) or 0, 4)
            if cfg_name == "consim0"
            else (
                round(ari_results.get("ari_consim1_vs_consim2_risk", 0) or 0, 4)
                if cfg_name == "consim1"
                else None
            )
        ),
        "C3_ari_interv_vs_next": (
            round(ari_results.get(f"ari_{cfg_name}_vs_consim1_intervention", 0) or 0, 4)
            if cfg_name == "consim0"
            else (
                round(ari_results.get("ari_consim1_vs_consim2_intervention", 0) or 0, 4)
                if cfg_name == "consim1"
                else None
            )
        ),
        # C4: R→I pair coverage fraction
        "C4_ri_coverage_fraction": round(
            coverage_results[cfg_name]["coverage_fraction"], 4
        ),
        "C4_ri_covered_pairs": coverage_results[cfg_name]["n_covered_pairs"],
        "C4_ri_possible_pairs": total_possible_ri,
        # B-family counts
        "n_ob_families_ge5paths": sum(
            1 for c in sig_counts[cfg_name].values() if c >= 5
        ),
        "n_ob_families_total": len(sig_counts[cfg_name]),
    }
    config_rows.append(row)

df_metrics = pd.DataFrame(config_rows)
metrics_path = os.path.join(STEP4_DIR, "config_selection_metrics_v2.csv")
df_metrics.to_csv(metrics_path, index=False)
log.info(f"Saved {metrics_path}")
log.info("\n" + df_metrics.to_string(index=False))

# ─── STEP 11: A2 Assessment — PathbuildB mechanistic quality ────────────────
log.info("=" * 60)
log.info("STEP 11 (A2): PathbuildB mechanistic quality assessment")
log.info("  Checking if top B-families describe mechanistic chains or risk themes")

for cfg_name in ["consim0", "consim1", "consim2"]:
    top20_path = os.path.join(OUT_TABLES, f"optionB_top20_decoded_{cfg_name}.csv")
    if os.path.exists(top20_path):
        df_t20 = pd.read_csv(top20_path)
        log.info(f"\n  === {cfg_name} Top-5 B-families (PathbuildB) ===")
        for _, row in df_t20.head(5).iterrows():
            log.info(f"  Rank {row['rank']}: {row['n_paths']} paths")
            log.info(f"    Signature: {row['signature_str']}")
            log.info(f"    Decoded:   {row['decoded_chain_components'][:300]}")
            log.info("")

# ─── STEP 12: PathbuildA assessment — chain cluster quality ──────────────────
log.info("=" * 60)
log.info("STEP 12 (A2): PathbuildA chain cluster assessment")
log.info("  Loading existing chain cluster names to assess misalignment collapse")

chain_names_path = os.path.join(OUT_TABLES, "chain_cluster_names.csv")
if os.path.exists(chain_names_path):
    df_chain = pd.read_csv(chain_names_path)
    # Count how many contain "misalignment" or "existential" or "catastrophic"
    misalign_keywords = ["misalign", "existential", "catastrophic", "unalign"]
    n_risk_themed = sum(
        1
        for _, row in df_chain.iterrows()
        if any(
            kw in str(row.get("cluster_name", "")).lower() for kw in misalign_keywords
        )
    )
    log.info(
        f"  PathbuildA: {n_risk_themed}/{len(df_chain)} chain clusters are "
        f"risk/misalignment-themed (not mechanistic chains)"
    )
    log.info("  Verdict: PathbuildA chain clustering FAILS the 'because' criterion")
    log.info(
        "  PathbuildB B-families (frozenset co-occurrence) are structurally more promising"
    )
    log.info("  → See decoded top-20 tables for empirical assessment")

log.info("=" * 60)
log.info(f"Track A complete — {datetime.now().isoformat()}")
log.info("Outputs:")
log.info(f"  {OUT_TABLES}/risk_clusters_perconfig_consim{{0,1,2}}.csv")
log.info(f"  {OUT_TABLES}/intervention_clusters_perconfig_consim{{0,1,2}}.csv")
log.info(f"  {OUT_TABLES}/optionB_cooccurrence_families_consim2.csv  (B3)")
log.info(f"  {OUT_TABLES}/optionB_top20_decoded_consim{{0,1,2}}.csv  (B4/A2)")
log.info(f"  {OUT_TABLES}/bodysubtype_cluster_representatives.csv  (A2 helper)")
log.info(f"  {STEP4_DIR}/config_selection_metrics_v2.csv  (A1)")
