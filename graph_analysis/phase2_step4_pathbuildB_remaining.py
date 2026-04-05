"""
Phase 2 Step 4 — Option B co-occurrence families for consim0 and consim1.

consim0 (edge-only): reads paths_unconstrained_edge_only.jsonl (uses 'path' key)
consim1 (VarA, max_consec_SIM<=1): reads representative_pathways_consim1.jsonl
         (uses 'node_id_sequence' key)

Output:
  step4_cluster_tables/optionB_cooccurrence_families_consim0.csv
  step4_cluster_tables/optionB_cooccurrence_families_consim1.csv
"""

import pickle
import json
import logging
import os
import sys
import time
from collections import defaultdict, Counter
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
OUT_PATHS = os.path.join(STEP4_DIR, "step4_paths")

for d in [OUT_TABLES, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "phase4_pathbuildB_remaining.log")
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
log.info("Phase 2 Step 4 — Option B remaining configs (consim0, consim1)")
log.info(f"Start: {datetime.now().isoformat()}")

# ─── Load PKL files ───────────────────────────────────────────────────────────
log.info("Loading PKL files …")
t0 = time.time()
with open(os.path.join(STEP1_DIR, "cluster_memberships.pkl"), "rb") as f:
    cm = pickle.load(f)
log.info(f"  cm: {len(cm)} keys  ({time.time() - t0:.1f}s)")

t1 = time.time()
with open(os.path.join(STEP1_DIR, "graph_node_attributes.pkl"), "rb") as f:
    node_attrs = pickle.load(f)
log.info(f"  node_attrs: {len(node_attrs)} nodes  ({time.time() - t1:.1f}s)")


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


# ─── Build valid_pathway_nodes (from paths_unconstrained_sim0.9.jsonl) ────────
log.info("Building valid_pathway_nodes from paths_unconstrained_sim0.9.jsonl …")
t_vp = time.time()
valid_pathway_nodes = set()
vp_file = os.path.join(PATHS_DIR, "paths_unconstrained_sim0.9.jsonl")
with open(vp_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        for nid in obj["path"]:
            valid_pathway_nodes.add(int(nid))
log.info(
    f"  {len(valid_pathway_nodes):,} valid-pathway nodes  ({time.time() - t_vp:.1f}s)"
)

# ─── Build node_to_stc mapping (body subtypes, valid_pathway_nodes-filtered) ─
BODY_SUBTYPES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]
log.info("Building node_to_stc mapping …")
t_stc = time.time()
node_to_stc = {}
for subtype in BODY_SUBTYPES:
    for cid, node_ids in get_clusters("0.9", "unconstrained", subtype).items():
        for nid in node_ids:
            if nid in valid_pathway_nodes:
                node_to_stc[nid] = (subtype, cid)
log.info(
    f"  {len(node_to_stc):,} nodes mapped (valid_pathway_nodes-filtered)  ({time.time() - t_stc:.1f}s)"
)


# ─── Option B computation function ───────────────────────────────────────────
def run_option_b(path_iter_fn, label, out_csv):
    """
    path_iter_fn: callable that yields lists of int node IDs (one per path)
    label: config name for logging
    out_csv: output file path
    """
    log.info(f"  [{label}] Pass 1: counting signatures …")
    t1 = time.time()
    sig_counts = Counter()
    n_paths = 0
    for path in path_iter_fn():
        if len(path) < 3:
            continue
        n_paths += 1
        body = path[1:-1]
        sig_parts = frozenset(node_to_stc[n] for n in body if n in node_to_stc)
        if sig_parts:
            sig_counts[sig_parts] += 1
    log.info(
        f"  [{label}] {n_paths:,} paths processed, {len(sig_counts):,} unique signatures  ({time.time() - t1:.1f}s)"
    )

    large_sigs_set = {s for s, c in sig_counts.items() if c >= 5}
    log.info(f"  [{label}] {len(large_sigs_set):,} families with n_paths>=5")

    log.info(f"  [{label}] Pass 2: collecting paths for large families …")
    t2 = time.time()
    sig_to_paths = defaultdict(list)
    for path in path_iter_fn():
        if len(path) < 3:
            continue
        body = path[1:-1]
        sig_parts = frozenset(node_to_stc[n] for n in body if n in node_to_stc)
        if sig_parts in large_sigs_set:
            sig_to_paths[sig_parts].append(path)
    log.info(
        f"  [{label}] {len(sig_to_paths):,} families collected  ({time.time() - t2:.1f}s)"
    )

    rows = []
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
        rows.append(
            {
                "family_id": fid,
                "n_paths": len(paths_list),
                "n_sources": n_src,
                "signature_str": sig_str[:200],
                "top_subtypes": str(dict(Counter(s[0] for s in sig).most_common(3))),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log.info(f"  [{label}] Saved {out_csv}  ({len(df)} rows)")

    # Print summary statistics
    if len(df) > 0:
        log.info(
            f"  [{label}] n_paths distribution: "
            f"min={df.n_paths.min()}, median={df.n_paths.median():.0f}, "
            f"max={df.n_paths.max()}, mean={df.n_paths.mean():.1f}"
        )
        log.info(
            f"  [{label}] n_sources distribution: "
            f"min={df.n_sources.min()}, median={df.n_sources.median():.0f}, "
            f"max={df.n_sources.max()}"
        )
        top5 = df.head(5)[
            ["family_id", "n_paths", "n_sources", "signature_str"]
        ].to_string(index=False)
        log.info(f"  [{label}] Top 5 families:\n{top5}")
    return df


# ─── CONSIM0: edge-only paths (uses 'path' key) ───────────────────────────────
log.info("=" * 50)
log.info("CONSIM0: Option B from paths_unconstrained_edge_only.jsonl")

eo_file = os.path.join(PATHS_DIR, "paths_unconstrained_edge_only.jsonl")


def iter_consim0():
    with open(eo_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            yield [int(x) for x in obj["path"]]


out_consim0 = os.path.join(OUT_TABLES, "optionB_cooccurrence_families_consim0.csv")
df_consim0 = run_option_b(iter_consim0, "consim0", out_consim0)

# ─── CONSIM1: representative_pathways_consim1.jsonl (uses 'node_id_sequence' key) ──
log.info("=" * 50)
log.info("CONSIM1: Option B from representative_pathways_consim1.jsonl")

c1_file = os.path.join(OUT_PATHS, "representative_pathways_consim1.jsonl")


def iter_consim1():
    with open(c1_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            yield [int(x) for x in obj["node_id_sequence"]]


out_consim1 = os.path.join(OUT_TABLES, "optionB_cooccurrence_families_consim1.csv")
df_consim1 = run_option_b(iter_consim1, "consim1", out_consim1)

# ─── Final summary ────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("SUMMARY")
log.info(f"  consim0: {len(df_consim0)} families — {out_consim0}")
log.info(f"  consim1: {len(df_consim1)} families — {out_consim1}")
log.info(f"End: {datetime.now().isoformat()}")
