"""
Phase 2 Step 4 — PathbuildB Connectivity (substep 27)
Computes R_cluster → B_family → I_cluster connectivity for all 3 pathbuildB configs:
  - consim0: edge-only paths
  - consim1: paths with max_consec_sim <= 1
  - consim2: all unconstrained paths (sim0.9 file, no SIM count filter)
"""

import pickle
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "phase2_results")
STEP1_DIR = os.path.join(RESULTS_DIR, "step1_load_and_parse_umapwithoutlocalsatellites")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4_finalanalysis")
PATHS_DIR = os.path.join(ROOT, "phase1_rawpathsfiles")
LOG_DIR = os.path.join(ROOT, "logfiles", "phase4_logs")
OUT_TABLES = os.path.join(STEP4_DIR, "step4_cluster_tables")
OUT_CONN = os.path.join(STEP4_DIR, "step4_connectivity")

for d in [OUT_CONN, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "phase4_pathbuildB_connectivity.log")
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
log.info("Phase 2 Step 4 — PathbuildB Connectivity (substep 27)")
log.info(f"Start: {datetime.now().isoformat()}")

# ─── Helpers ──────────────────────────────────────────────────────────────────
BODY_SUBTYPES = {
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
}

SUBTYPE_PREFIX = {
    "problem_analysis": "pr",
    "theoretical_insight": "th",
    "design_rationale": "de",
    "implementation_mechanism": "im",
    "validation_evidence": "va",
}


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def max_consec_sim(path_ids, sim_edge_set):
    max_run = run = 0
    for i in range(len(path_ids) - 1):
        a, b = int(path_ids[i]), int(path_ids[i + 1])
        if (min(a, b), max(a, b)) in sim_edge_set:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def sig_from_frozenset(fs):
    """Convert frozenset of (subtype, cluster_id) to canonical signature_str."""
    parts = sorted(f"{SUBTYPE_PREFIX.get(s, s[:2])}:{cid}" for s, cid in fs)
    return " & ".join(parts)


def counter_to_df(counter):
    rows = [
        {"cluster_a": k[0], "cluster_b": k[1], "n_paths": v} for k, v in counter.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("n_paths", ascending=False)
        .reset_index(drop=True)
    )


def get_clusters(cm, edge_config, mode, node_type, algo="agglomerative"):
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


# ─── Load PKL files ───────────────────────────────────────────────────────────
log.info("Loading cluster_memberships.pkl …")
t0 = time.time()
with open(os.path.join(STEP1_DIR, "cluster_memberships.pkl"), "rb") as f:
    cm = pickle.load(f)
log.info(f"  {len(cm)} keys  ({time.time() - t0:.1f}s)")

log.info("Loading graph_node_attributes.pkl …")
t1 = time.time()
with open(os.path.join(STEP1_DIR, "graph_node_attributes.pkl"), "rb") as f:
    node_attrs = pickle.load(f)
log.info(f"  {len(node_attrs)} nodes  ({time.time() - t1:.1f}s)")

log.info("Loading graph_edge_data.pkl (needed for sim_edge_set) …")
t2 = time.time()
with open(os.path.join(STEP1_DIR, "graph_edge_data.pkl"), "rb") as f:
    edge_data = pickle.load(f)
log.info(f"  {len(edge_data)} edges  ({time.time() - t2:.1f}s)")

# ─── Build broad unconstrained VPN first (needed to restrict sim_edge_set) ────
# maturity>=3 endpoint filter — path gen used ALL_INTERVENTION_IDS
# sim_edge_set is config-agnostic; restrict it to the broadest VPN (unconstrained)
log.info("Building unconstrained valid_pathway_nodes for sim_edge_set restriction …")
t_vpn_broad = time.time()
vpn_unconstrained = set()
_paths_file_broad = os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl")
with open(_paths_file_broad) as _f:
    for _line in _f:
        _obj = json.loads(_line)
        _path = [int(x) for x in _obj["path"]]
        _interv_id = _path[-1]
        if (
            int(node_attrs.get(_interv_id, {}).get("intervention_maturity", 0) or 0)
            >= 3
        ):
            vpn_unconstrained.update(_path)
log.info(
    f"  {len(vpn_unconstrained)} unconstrained VPN nodes  ({time.time() - t_vpn_broad:.1f}s)"
)

# ─── Build sim_edge_set (SIM >= 0.9, restricted to unconstrained VPN pairs) ───
log.info("Building sim_edge_set (SIM>=0.9, VPN-restricted) …")
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
log.info(f"  {len(sim_edge_set)} SIM>=0.9 pairs  ({time.time() - t_sim:.1f}s)")

# ─── Build node_to_stc (body nodes → (subtype, cluster_id)) ───────────────────
log.info("Building node_to_stc from cluster_memberships …")
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
            node_to_stc[int(nid)] = (str(nt), str(cid))
log.info(f"  node_to_stc: {len(node_to_stc)} body nodes mapped")

# ─── Load risk and intervention clusters (base — will be filtered per config) ──
log.info(
    "Loading base risk/intervention clusters (SIM=0.9, unconstrained, agglomerative) …"
)

# Risk: use the risk_clusters_09.pkl if available for consistency, else rebuild
risk_clusters_09_pkl = os.path.join(STEP4_DIR, "risk_clusters_09.pkl")
if os.path.exists(risk_clusters_09_pkl):
    with open(risk_clusters_09_pkl, "rb") as f:
        risk_clusters_base = pickle.load(f)
    log.info(f"  risk_clusters: loaded from PKL ({len(risk_clusters_base)} clusters)")
else:
    risk_clusters_base = get_clusters(cm, "0.9", "unconstrained", "risk")
    log.info(f"  risk_clusters: built from cm ({len(risk_clusters_base)} clusters)")

interv_clusters_base = get_clusters(cm, "0.9", "unconstrained", "intervention")
log.info(f"  interv_clusters: {len(interv_clusters_base)} clusters")

# ─── Load family signature lookup tables ──────────────────────────────────────
log.info("Loading family CSV files …")

FAMILY_FILES = {
    "consim0": os.path.join(OUT_TABLES, "optionB_cooccurrence_families_consim0.csv"),
    "consim1": os.path.join(OUT_TABLES, "optionB_cooccurrence_families_consim1.csv"),
    "consim2": os.path.join(OUT_TABLES, "optionB_cooccurrence_families.csv"),
}

sig_to_family = {}
all_family_ids = {}
for config, fpath in FAMILY_FILES.items():
    df = pd.read_csv(fpath)
    sig_to_family[config] = dict(zip(df["signature_str"], df["family_id"].astype(str)))
    all_family_ids[config] = set(df["family_id"].astype(str))
    log.info(f"  {config}: {len(df)} families from {fpath}")

# ─── Config definitions ────────────────────────────────────────────────────────
CONFIGS = [
    {
        "name": "consim0",
        "paths_file": os.path.join(PATHS_DIR, "paths_unconstrained_edge_only.jsonl"),
        "filter_fn": None,  # no SIM filter for edge-only
    },
    {
        "name": "consim1",
        "paths_file": os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl"),
        "filter_fn": lambda path: max_consec_sim(path, sim_edge_set) <= 1,
    },
    {
        "name": "consim2",
        "paths_file": os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl"),
        "filter_fn": None,  # no SIM count filter
    },
]

# ─── Per-config processing ─────────────────────────────────────────────────────
for cfg in CONFIGS:
    config_name = cfg["name"]
    paths_file = cfg["paths_file"]
    filter_fn = cfg["filter_fn"]
    log.info("=" * 60)
    log.info(f"Processing config: {config_name}")

    # Build valid_pathway_nodes for this config
    # maturity>=3 endpoint filter — path gen used ALL_INTERVENTION_IDS
    log.info(f"  Building valid_pathway_nodes from {os.path.basename(paths_file)} …")
    t_vp = time.time()
    valid_pathway_nodes = set()
    n_paths_vpn = 0
    with open(paths_file) as f:
        for line in f:
            obj = json.loads(line)
            path = [int(x) for x in obj["path"]]
            # Apply consim filter even for valid_pathway_nodes if needed
            if filter_fn is not None and not filter_fn(path):
                continue
            interv_id = path[-1]
            if (
                int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0)
                >= 3
            ):
                valid_pathway_nodes.update(path)
                n_paths_vpn += 1
    log.info(
        f"    {len(valid_pathway_nodes)} valid-pathway nodes from {n_paths_vpn} paths  ({time.time() - t_vp:.1f}s)"
    )

    # Filter risk clusters to valid_pathway_nodes
    risk_clusters = {
        cid: [n for n in nodes if n in valid_pathway_nodes]
        for cid, nodes in risk_clusters_base.items()
    }
    risk_clusters = {cid: nodes for cid, nodes in risk_clusters.items() if nodes}

    # Filter intervention clusters to valid_pathway_nodes + maturity >= 3
    interv_clusters = {
        cid: [
            n
            for n in nodes
            if n in valid_pathway_nodes
            and int(node_attrs.get(n, {}).get("intervention_maturity", 0) or 0) >= 3
        ]
        for cid, nodes in interv_clusters_base.items()
    }
    interv_clusters = {cid: nodes for cid, nodes in interv_clusters.items() if nodes}

    # Build node → cluster mappings
    node_to_risk = {}
    for cid, node_ids in risk_clusters.items():
        for nid in node_ids:
            node_to_risk[nid] = str(cid)

    node_to_interv = {}
    for cid, node_ids in interv_clusters.items():
        for nid in node_ids:
            node_to_interv[nid] = str(cid)

    log.info(
        f"    risk clusters: {len(risk_clusters)}, node_to_risk: {len(node_to_risk)}"
    )
    log.info(
        f"    interv clusters: {len(interv_clusters)}, node_to_interv: {len(node_to_interv)}"
    )

    # Load family sig→id for this config
    s2f = sig_to_family[config_name]
    all_fam_ids = all_family_ids[config_name]

    # Streaming pass over paths
    risk_to_family: Counter = Counter()
    family_to_interv: Counter = Counter()
    risk_to_interv: Counter = Counter()

    n_paths_read = 0
    n_paths_kept = 0
    n_no_cluster = 0
    n_no_family = 0
    family_match_counter: Counter = Counter()

    log.info(f"  Streaming paths from {os.path.basename(paths_file)} …")
    t_stream = time.time()

    with open(paths_file) as f:
        for line in f:
            obj = json.loads(line)
            path = [int(x) for x in obj["path"]]
            n_paths_read += 1

            if filter_fn is not None and not filter_fn(path):
                continue
            n_paths_kept += 1

            risk_node = path[0]
            interv_node = path[-1]
            risk_cid = node_to_risk.get(risk_node)
            interv_cid = node_to_interv.get(interv_node)

            if risk_cid is None or interv_cid is None:
                n_no_cluster += 1
                continue

            # Always count risk_to_interv
            risk_to_interv[(risk_cid, interv_cid)] += 1

            # Compute family signature from body nodes
            body_ids = path[1:-1]
            sig_parts = frozenset(
                node_to_stc[nid] for nid in body_ids if nid in node_to_stc
            )
            if not sig_parts:
                n_no_family += 1
                continue

            sig_str = sig_from_frozenset(sig_parts)
            family_id = s2f.get(sig_str)

            if family_id is None:
                n_no_family += 1
                continue

            family_match_counter[family_id] += 1
            risk_to_family[(risk_cid, family_id)] += 1
            family_to_interv[(family_id, interv_cid)] += 1

    log.info(f"  Done in {time.time() - t_stream:.1f}s")
    log.info(
        f"  paths_read={n_paths_read:,}  kept={n_paths_kept:,}  no_cluster={n_no_cluster:,}  no_family={n_no_family:,}"
    )
    log.info(f"  risk_to_family edges: {len(risk_to_family)}")
    log.info(f"  family_to_interv edges: {len(family_to_interv)}")
    log.info(f"  risk_to_interv edges: {len(risk_to_interv)}")
    log.info(f"  families matched: {len(family_match_counter)} / {len(all_fam_ids)}")

    # Save edge CSVs
    suffix = f"_{config_name}"

    r2f_df = counter_to_df(risk_to_family)
    r2f_path = os.path.join(OUT_CONN, f"risk_to_Bfamily_edges{suffix}.csv")
    r2f_df.to_csv(r2f_path, index=False)
    log.info(f"  Saved {os.path.basename(r2f_path)} ({len(r2f_df)} rows)")

    f2i_df = counter_to_df(family_to_interv)
    f2i_path = os.path.join(OUT_CONN, f"Bfamily_to_interv_edges{suffix}.csv")
    f2i_df.to_csv(f2i_path, index=False)
    log.info(f"  Saved {os.path.basename(f2i_path)} ({len(f2i_df)} rows)")

    r2i_df = counter_to_df(risk_to_interv)
    r2i_path = os.path.join(OUT_CONN, f"risk_to_interv_via_B_edges{suffix}.csv")
    r2i_df.to_csv(r2i_path, index=False)
    log.info(f"  Saved {os.path.basename(r2i_path)} ({len(r2i_df)} rows)")

    # Gap analysis
    all_risk_set = set(str(c) for c in risk_clusters.keys())
    all_interv_set = set(str(c) for c in interv_clusters.keys())

    risk_with_family = (
        set(r2f_df["cluster_a"].astype(str)) if len(r2f_df) > 0 else set()
    )
    family_with_risk = (
        set(r2f_df["cluster_b"].astype(str)) if len(r2f_df) > 0 else set()
    )
    family_with_interv = (
        set(f2i_df["cluster_a"].astype(str)) if len(f2i_df) > 0 else set()
    )
    interv_with_family = (
        set(f2i_df["cluster_b"].astype(str)) if len(f2i_df) > 0 else set()
    )
    risk_with_interv_direct = (
        set(r2i_df["cluster_a"].astype(str)) if len(r2i_df) > 0 else set()
    )
    interv_with_risk_direct = (
        set(r2i_df["cluster_b"].astype(str)) if len(r2i_df) > 0 else set()
    )

    gap_rows = [
        {
            "gap_type": "risk_clusters_with_no_Bfamily_connection",
            "count": len(all_risk_set - risk_with_family),
            "examples": ",".join(sorted(all_risk_set - risk_with_family)[:5]),
        },
        {
            "gap_type": "Bfamilies_with_no_risk_connection",
            "count": len(all_fam_ids - family_with_risk),
            "examples": ",".join(sorted(all_fam_ids - family_with_risk)[:5]),
        },
        {
            "gap_type": "Bfamilies_with_no_interv_connection",
            "count": len(all_fam_ids - family_with_interv),
            "examples": ",".join(sorted(all_fam_ids - family_with_interv)[:5]),
        },
        {
            "gap_type": "interv_clusters_with_no_Bfamily_connection",
            "count": len(all_interv_set - interv_with_family),
            "examples": ",".join(sorted(all_interv_set - interv_with_family)[:5]),
        },
        {
            "gap_type": "risk_clusters_with_no_direct_interv_link",
            "count": len(all_risk_set - risk_with_interv_direct),
            "examples": ",".join(sorted(all_risk_set - risk_with_interv_direct)[:5]),
        },
        {
            "gap_type": "interv_clusters_with_no_direct_risk_link",
            "count": len(all_interv_set - interv_with_risk_direct),
            "examples": ",".join(sorted(all_interv_set - interv_with_risk_direct)[:5]),
        },
    ]
    gap_path = os.path.join(OUT_CONN, f"gap_analysis_pathbuildB{suffix}.csv")
    pd.DataFrame(gap_rows).to_csv(gap_path, index=False)
    log.info(f"  Saved {os.path.basename(gap_path)}")
    for g in gap_rows:
        log.info(f"    {g['gap_type']}: {g['count']}")

log.info("=" * 70)
log.info(f"PathbuildB Connectivity COMPLETE — {datetime.now().isoformat()}")
