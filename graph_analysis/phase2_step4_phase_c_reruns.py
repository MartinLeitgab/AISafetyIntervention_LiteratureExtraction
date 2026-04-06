"""
Phase 2 Step 4 — Phase C Reruns
Recomputes all Category B Step 2/3 analyses restricted to valid_pathway_nodes.
Writes all outputs to step4_finalanalysis/ with _qualifying suffix.

Analyses (plan Phase C items 20-28):
  20. Source diversity per cluster (v2 — qualifying)
  21. Maturity distribution per cluster (qualifying)
  22. Multi-risk cluster analysis (qualifying)
  23. Risk diversity stats (qualifying)
  24. Mechanism family categorization (qualifying)
  25. Source diversity v1 per cluster (qualifying)
  26. Temporal coverage per cluster (qualifying)
 26b. Lifecycle distribution per cluster (qualifying)
  27. SIM coverage: anchored vs SIM-only (consim0 vs consim2 path membership)
  28. Held-out embedding validation on qualifying cluster members

All item 19 (hub quality) is already done by fix scripts — skip.
"""

import pickle
import json
import os
import sys
import time
import logging
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.linalg import norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "phase2_results")
STEP1_DIR = os.path.join(RESULTS_DIR, "step1_load_and_parse_umapwithoutlocalsatellites")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4_finalanalysis")
PATHS_DIR = os.path.join(ROOT, "phase1_rawpathsfiles")
STEP4_PATHS = os.path.join(STEP4_DIR, "step4_paths")
LOG_DIR = os.path.join(ROOT, "logfiles", "phase4_logs")

os.makedirs(STEP4_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "phase4_phase_c_reruns.log")
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
log.info("Phase 2 Step 4 — Phase C Category B Reruns (valid_pathway_nodes)")
log.info(f"Start: {datetime.now().isoformat()}")

# ─── Load PKL files ────────────────────────────────────────────────────────────
log.info("Loading PKL files …")
t0 = time.time()
with open(os.path.join(STEP1_DIR, "cluster_memberships.pkl"), "rb") as f:
    cm = pickle.load(f)
with open(os.path.join(STEP1_DIR, "graph_node_attributes.pkl"), "rb") as f:
    node_attrs = pickle.load(f)
log.info(
    f"  cm: {len(cm)} keys, node_attrs: {len(node_attrs)} nodes  ({time.time() - t0:.1f}s)"
)


# ─── Cluster helper ────────────────────────────────────────────────────────────
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


# ─── Load valid_pathway_nodes (unconstrained) ─────────────────────────────────
# maturity>=3 endpoint filter — path gen used ALL_INTERVENTION_IDS
log.info("Loading valid_pathway_nodes (unconstrained) …")
t_vp = time.time()
valid_pathway_nodes = set()
paths_file = os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl")
with open(paths_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) >= 3:
            valid_pathway_nodes.update(path)
log.info(
    f"  {len(valid_pathway_nodes)} valid-pathway nodes  ({time.time() - t_vp:.1f}s)"
)

# ─── Load consim0 and consim2 path memberships (for SIM coverage analysis) ───
log.info("Loading consim0 and consim2 node sets …")
consim0_nodes = set()
eo_file = os.path.join(PATHS_DIR, "paths_unconstrained_edge_only.jsonl")
with open(eo_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        for nid in obj["path"]:
            consim0_nodes.add(int(nid))
log.info(f"  consim0 nodes: {len(consim0_nodes)}")

# consim2 = all VarB paths (max_consec_SIM <= 2) — these are in the output path file
consim2_nodes = set()
consim2_file = os.path.join(STEP4_PATHS, "representative_pathways_consim2.jsonl")
if os.path.exists(consim2_file):
    with open(consim2_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            for nid in obj.get("node_id_sequence", []):
                consim2_nodes.add(int(nid))
    log.info(f"  consim2 nodes: {len(consim2_nodes)}")
else:
    log.warning("  consim2 path file not found — skipping SIM coverage analysis")
    consim2_nodes = None

# ─── SECTION 20: Source diversity v2 per cluster (qualifying) ─────────────────
log.info("=" * 50)
log.info("SECTION 20: Source diversity v2 per cluster (qualifying)")

NODE_TYPES = [
    "risk",
    "intervention",
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]
EC = "0.9"
MODE = "unconstrained"

rows_sd = []
for nt in NODE_TYPES:
    for cid, raw_ids in get_clusters(EC, MODE, nt).items():
        node_ids = [n for n in raw_ids if n in valid_pathway_nodes]
        if not node_ids:
            continue
        urls = set(str(node_attrs.get(n, {}).get("url", "")) for n in node_ids) - {
            "",
            "None",
            "nan",
        }
        rows_sd.append(
            {
                "edge_config": EC,
                "mode": MODE,
                "node_type": nt,
                "cluster_id": cid,
                "n_sources": len(urls),
                "cluster_size": len(node_ids),
                "nodes_with_sources": sum(
                    1 for n in node_ids if node_attrs.get(n, {}).get("url", "")
                ),
                "source_coverage_pct": round(100 * len(urls) / len(node_ids), 1)
                if node_ids
                else 0,
            }
        )

sd_df = pd.DataFrame(rows_sd)
out_sd = os.path.join(STEP4_DIR, "source_diversity_qualifying.csv")
sd_df.to_csv(out_sd, index=False)
log.info(f"  Saved {out_sd} ({len(sd_df)} rows)")

# ─── SECTION 25: Source diversity v1 (qualifying) ─────────────────────────────
log.info("=" * 50)
log.info("SECTION 25: Source diversity v1 (qualifying)")

rows_sd1 = []
for nt in ["risk", "intervention"]:
    for cid, raw_ids in get_clusters(EC, MODE, nt).items():
        node_ids = [n for n in raw_ids if n in valid_pathway_nodes]
        if not node_ids:
            continue
        urls = set(str(node_attrs.get(n, {}).get("url", "")) for n in node_ids) - {
            "",
            "None",
            "nan",
        }
        rows_sd1.append(
            {
                "edge_config": EC,
                "mode": MODE,
                "node_type": nt,
                "cluster_id": cid,
                "n_sources": len(urls),
                "cluster_size": len(node_ids),
                "nodes_with_sources": sum(
                    1 for n in node_ids if node_attrs.get(n, {}).get("url", "")
                ),
            }
        )

sd1_df = pd.DataFrame(rows_sd1)
out_sd1 = os.path.join(STEP4_DIR, "source_diversity_v1_qualifying.csv")
sd1_df.to_csv(out_sd1, index=False)
log.info(f"  Saved {out_sd1} ({len(sd1_df)} rows)")

# ─── SECTION 21: Maturity distribution per cluster (qualifying) ───────────────
log.info("=" * 50)
log.info("SECTION 21: Maturity distribution per cluster (qualifying)")

mat_data = defaultdict(Counter)
for cid, raw_ids in get_clusters(EC, MODE, "intervention").items():
    node_ids = [n for n in raw_ids if n in valid_pathway_nodes]
    for nid in node_ids:
        mat = node_attrs.get(nid, {}).get("intervention_maturity")
        if mat is not None:
            try:
                mat_data[str(cid)][int(mat)] += 1
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
    ax.set_title(
        "Intervention Cluster × Maturity Distribution (valid_pathway_nodes-filtered)"
    )
    plt.colorbar(im, ax=ax, label="Proportion")
    plt.tight_layout()
    out_mat = os.path.join(STEP4_DIR, "maturity_distribution_qualifying.png")
    plt.savefig(out_mat, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out_mat}")

    # Also save CSV
    mat_rows = []
    for cid in cids_s:
        total = sum(mat_data[cid].values())
        row = {"cluster_id": cid, "total_qualifying": total}
        for lv in [1, 2, 3, 4]:
            row[f"maturity_{lv}_count"] = mat_data[cid].get(lv, 0)
            row[f"maturity_{lv}_pct"] = (
                round(100 * mat_data[cid].get(lv, 0) / total, 1) if total else 0
            )
        mat_rows.append(row)
    pd.DataFrame(mat_rows).to_csv(
        os.path.join(STEP4_DIR, "maturity_per_cluster_qualifying.csv"), index=False
    )
    log.info("  Saved maturity_per_cluster_qualifying.csv")

# ─── SECTION 22: Multi-risk cluster analysis (qualifying) ─────────────────────
log.info("=" * 50)
log.info("SECTION 22: Multi-risk cluster analysis (qualifying)")

rows_mr = []
for cid, raw_ids in get_clusters(EC, MODE, "risk").items():
    node_ids = [n for n in raw_ids if n in valid_pathway_nodes]
    if not node_ids:
        continue
    cats = [node_attrs.get(n, {}).get("concept_category", "") for n in node_ids]
    unique_cats = set(c for c in cats if c)
    rows_mr.append(
        {
            "edge_config": EC,
            "mode": MODE,
            "cluster_id": cid,
            "cluster_size": len(node_ids),
            "n_unique_risk_categories": len(unique_cats),
            "is_multi_risk": len(unique_cats) > 1,
            "categories": "|".join(sorted(unique_cats)),
        }
    )

mr_df = pd.DataFrame(rows_mr)
out_mr = os.path.join(STEP4_DIR, "multi_risk_clusters_qualifying.csv")
mr_df.to_csv(out_mr, index=False)
n_multi = mr_df["is_multi_risk"].sum()
log.info(f"  Saved {out_mr} ({len(mr_df)} rows, {n_multi} multi-risk clusters)")

# ─── SECTION 23: Risk diversity stats (qualifying) ────────────────────────────
log.info("=" * 50)
log.info("SECTION 23: Risk diversity stats (qualifying)")

all_risk_qualifying = []
for cid, raw_ids in get_clusters(EC, MODE, "risk").items():
    all_risk_qualifying.extend([n for n in raw_ids if n in valid_pathway_nodes])

# Unique risk names and categories
risk_cats = [
    node_attrs.get(n, {}).get("concept_category", "risk") for n in all_risk_qualifying
]
cat_counts = Counter(risk_cats)
total_r = len(all_risk_qualifying)
top_cat = cat_counts.most_common(1)[0] if cat_counts else ("risk", total_r)

# Gini coefficient of cluster sizes
risk_cluster_sizes = []
for cid, raw_ids in get_clusters(EC, MODE, "risk").items():
    node_ids = [n for n in raw_ids if n in valid_pathway_nodes]
    if node_ids:
        risk_cluster_sizes.append(len(node_ids))

sizes = np.array(sorted(risk_cluster_sizes))
n = len(sizes)
if n > 1:
    gini = (2 * np.sum(np.arange(1, n + 1) * sizes) - (n + 1) * np.sum(sizes)) / (
        n * np.sum(sizes)
    )
else:
    gini = 0.0

rd_row = {
    "edge_config": EC,
    "mode": MODE,
    "n_risk_nodes_total": total_r,
    "n_unique_categories": len(cat_counts),
    "gini_coefficient": round(float(gini), 4),
    "top_category": top_cat[0],
    "top_category_pct": round(100 * top_cat[1] / total_r, 1) if total_r else 0,
    "n_qualifying_clusters": len(risk_cluster_sizes),
    "mean_cluster_size": round(np.mean(risk_cluster_sizes), 1)
    if risk_cluster_sizes
    else 0,
}
pd.DataFrame([rd_row]).to_csv(
    os.path.join(STEP4_DIR, "risk_diversity_qualifying.csv"), index=False
)
log.info(
    f"  Risk diversity: {total_r} qualifying risk nodes, {len(cat_counts)} categories, Gini={gini:.4f}"
)

# ─── SECTION 24: Mechanism family categorization (qualifying) ─────────────────
log.info("=" * 50)
log.info("SECTION 24: Mechanism family categorization (qualifying)")

BODY_SUBTYPES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]

rows_mf = []
for subtype in BODY_SUBTYPES:
    all_for_type = []
    for cid, raw_ids in get_clusters(EC, MODE, subtype).items():
        node_ids = [n for n in raw_ids if n in valid_pathway_nodes]
        if not node_ids:
            continue
        exemplar_nid = node_ids[0]
        top5 = " | ".join(
            node_attrs.get(n, {}).get("name", str(n))[:60] for n in node_ids[:5]
        )
        rows_mf.append(
            {
                "edge_config": EC,
                "mode": MODE,
                "concept_category": subtype,
                "cluster_id": cid,
                "cluster_size": len(node_ids),
                "exemplar_name": node_attrs.get(exemplar_nid, {}).get("name", "")[:100],
                "top5_members": top5[:300],
                "n_sources": len(
                    set(str(node_attrs.get(n, {}).get("url", "")) for n in node_ids)
                    - {"", "None", "nan"}
                ),
            }
        )

mf_df = pd.DataFrame(rows_mf).sort_values(
    ["concept_category", "cluster_size"], ascending=[True, False]
)
out_mf = os.path.join(STEP4_DIR, "mechanism_families_qualifying.csv")
mf_df.to_csv(out_mf, index=False)
log.info(f"  Saved {out_mf} ({len(mf_df)} rows)")

# ─── SECTION 26: Temporal coverage per cluster (qualifying) ───────────────────
log.info("=" * 50)
log.info("SECTION 26: Temporal coverage per cluster (qualifying)")

rows_tc = []
for nt in ["risk", "intervention"]:
    for cid, raw_ids in get_clusters(EC, MODE, nt).items():
        node_ids = [n for n in raw_ids if n in valid_pathway_nodes]
        if not node_ids:
            continue
        years = []
        for nid in node_ids:
            yr = node_attrs.get(nid, {}).get("first_published")
            if yr is not None:
                try:
                    years.append(int(yr))
                except Exception:
                    pass
        if not years:
            continue
        rows_tc.append(
            {
                "edge_config": EC,
                "mode": MODE,
                "node_type": nt,
                "cluster_id": cid,
                "year_min": min(years),
                "year_max": max(years),
                "year_range": max(years) - min(years),
                "year_mean": round(float(np.mean(years)), 2),
                "year_median": float(np.median(years)),
                "n_nodes_with_year": len(years),
                "cluster_size": len(node_ids),
            }
        )

tc_df = pd.DataFrame(rows_tc)
out_tc = os.path.join(STEP4_DIR, "temporal_coverage_qualifying.csv")
tc_df.to_csv(out_tc, index=False)
log.info(f"  Saved {out_tc} ({len(tc_df)} rows)")

if not tc_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, nt in zip(axes, ["risk", "intervention"]):
        sub = tc_df[tc_df["node_type"] == nt].sort_values("cluster_id")
        if sub.empty:
            continue
        cids = sub["cluster_id"].tolist()
        ax.bar(
            range(len(cids)), sub["year_range"].tolist(), color="steelblue", alpha=0.7
        )
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels(cids, rotation=90, fontsize=6)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Year range (max - min)")
        ax.set_title(f"{nt.title()} clusters — temporal spread (qualifying members)")
    plt.suptitle("Temporal Coverage Per Cluster (valid_pathway_nodes-filtered)")
    plt.tight_layout()
    out_tcpng = os.path.join(STEP4_DIR, "temporal_coverage_qualifying.png")
    plt.savefig(out_tcpng, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out_tcpng}")

# ─── SECTION 26b: Lifecycle distribution per cluster (qualifying) ──────────────
log.info("=" * 50)
log.info("SECTION 26b: Lifecycle distribution per cluster (qualifying)")

lc_data = defaultdict(Counter)
for cid, raw_ids in get_clusters(EC, MODE, "intervention").items():
    node_ids = [n for n in raw_ids if n in valid_pathway_nodes]
    for nid in node_ids:
        lc = node_attrs.get(nid, {}).get("intervention_lifecycle")
        if lc is not None:
            try:
                lc_data[str(cid)][int(lc)] += 1
            except Exception:
                pass

if lc_data:
    cids_s = sorted(lc_data.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
    lc_m = np.zeros((6, len(cids_s)))
    for j, cid in enumerate(cids_s):
        total = sum(lc_data[cid].values())
        for i, lv in enumerate([1, 2, 3, 4, 5, 6]):
            lc_m[i, j] = lc_data[cid].get(lv, 0) / total if total else 0
    fig, ax = plt.subplots(figsize=(max(14, len(cids_s) // 2), 5))
    im = ax.imshow(lc_m, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cids_s)))
    ax.set_xticklabels(cids_s, rotation=90, fontsize=7)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"Lifecycle {lvl}" for lvl in [1, 2, 3, 4, 5, 6]], fontsize=9)
    ax.set_title(
        "Intervention Cluster × Lifecycle Distribution (valid_pathway_nodes-filtered)"
    )
    plt.colorbar(im, ax=ax, label="Proportion")
    plt.tight_layout()
    out_lc = os.path.join(STEP4_DIR, "lifecycle_distribution_qualifying.png")
    plt.savefig(out_lc, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out_lc}")

# ─── SECTION 27: SIM coverage — anchored vs SIM-only nodes ───────────────────
log.info("=" * 50)
log.info("SECTION 27: SIM coverage — anchored vs SIM-only nodes")

if consim2_nodes:
    rows_sim = []
    for nt in ["risk", "intervention"]:
        for cid, raw_ids in get_clusters(EC, MODE, nt).items():
            node_ids = [n for n in raw_ids if n in valid_pathway_nodes]
            if not node_ids:
                continue
            n_consim0 = sum(1 for n in node_ids if n in consim0_nodes)
            n_consim2 = sum(1 for n in node_ids if n in consim2_nodes)
            n_total = len(node_ids)
            rows_sim.append(
                {
                    "node_type": nt,
                    "cluster_id": cid,
                    "n_qualifying": n_total,
                    "n_consim0_anchored": n_consim0,
                    "n_consim2_only": n_consim2 - n_consim0,
                    "n_consim0_fraction": round(n_consim0 / n_total, 4)
                    if n_total
                    else 0,
                    "n_consim2_fraction": round(n_consim2 / n_total, 4)
                    if n_total
                    else 0,
                    "edge_only_path_fraction": round(n_consim0 / n_total, 4)
                    if n_total
                    else 0,
                }
            )
    sim_df = pd.DataFrame(rows_sim)
    out_sim = os.path.join(STEP4_DIR, "sim_coverage_qualifying.csv")
    sim_df.to_csv(out_sim, index=False)
    log.info(f"  Saved {out_sim} ({len(sim_df)} rows)")

    # Summary stats
    for nt in ["risk", "intervention"]:
        sub = sim_df[sim_df["node_type"] == nt]
        if not sub.empty:
            log.info(
                f"  {nt}: mean edge-only-path-fraction={sub['edge_only_path_fraction'].mean():.3f}, "
                f"mean consim2-fraction={sub['n_consim2_fraction'].mean():.3f}"
            )

# ─── SECTION 28: Held-out embedding validation (qualifying) ───────────────────
log.info("=" * 50)
log.info("SECTION 28: Held-out embedding validation on qualifying cluster members")


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


# Build embedding cache for qualifying nodes only (memory efficient)
log.info("  Building embedding cache for qualifying nodes …")
t_emb = time.time()
emb_cache = {}
for nid in valid_pathway_nodes:
    attrs = node_attrs.get(nid, {})
    emb_raw = attrs.get("embedding")
    if emb_raw is not None:
        try:
            emb_cache[nid] = parse_embedding(emb_raw)
        except Exception:
            pass
log.info(
    f"  emb_cache: {len(emb_cache)} qualifying nodes  ({time.time() - t_emb:.1f}s)"
)

N_SPLITS = 5
HOLDOUT_FRAC = 0.2
rng = np.random.default_rng(42)

rows_hv = []
for nt in ["risk", "intervention"]:
    for cid, raw_ids in get_clusters(EC, MODE, nt).items():
        node_ids = [n for n in raw_ids if n in valid_pathway_nodes and n in emb_cache]
        if len(node_ids) < 5:
            continue
        scores = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(len(node_ids))
            n_hold = max(1, int(len(node_ids) * HOLDOUT_FRAC))
            hold_idx = perm[:n_hold]
            train_idx = perm[n_hold:]
            if len(train_idx) < 2:
                continue
            centroid = np.mean([emb_cache[node_ids[i]] for i in train_idx], axis=0)
            sims = [cosine_sim(emb_cache[node_ids[i]], centroid) for i in hold_idx]
            scores.append(float(np.mean(sims)))
        if scores:
            rows_hv.append(
                {
                    "node_type": nt,
                    "cluster_id": cid,
                    "n_qualifying": len(node_ids),
                    "holdout_centroid_sim_mean": round(float(np.mean(scores)), 4),
                    "holdout_centroid_sim_std": round(float(np.std(scores)), 4),
                }
            )

hv_df = pd.DataFrame(rows_hv)
out_hv = os.path.join(STEP4_DIR, "held_out_validation_qualifying.csv")
hv_df.to_csv(out_hv, index=False)
log.info(f"  Saved {out_hv} ({len(hv_df)} rows)")
if not hv_df.empty:
    for nt in ["risk", "intervention"]:
        sub = hv_df[hv_df["node_type"] == nt]
        if not sub.empty:
            log.info(
                f"  {nt}: mean holdout-centroid-sim = {sub['holdout_centroid_sim_mean'].mean():.4f} "
                f"(range {sub['holdout_centroid_sim_mean'].min():.4f}–{sub['holdout_centroid_sim_mean'].max():.4f})"
            )

log.info("=" * 70)
log.info(f"Phase C COMPLETE — {datetime.now().isoformat()}")
