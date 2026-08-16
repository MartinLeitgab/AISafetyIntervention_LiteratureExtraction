"""
Phase 2 Step 5b — Workshop Pathway Examples

Extracts human-readable pathway examples for:
  1. Prevalent pathways — top-15 risk→intervention connections by n_paths,
     top-10 Option A chain clusters, top-10 Option B co-occurrence families
  2. Gap pathways — top-10 thin-coverage risk clusters (fewest total paths)

Reads LLM cluster names from step5_naming/ if available; falls back to
algorithmic names from step4_cluster_tables/.

Inputs:
  optionA_kmeans_model.pkl
  cluster_memberships.pkl
  graph_node_attributes.pkl
  step4_paths/representative_pathways_consim2.jsonl
  step4_paths/representative_pathways_edgeonly.jsonl
  step4_connectivity/risk_to_intervention_edges.csv
  step4_cluster_tables/optionB_cooccurrence_families.csv
  step5_naming/*.csv  (optional — used for LLM names if present)

Outputs (step5_examples/):
  pathway_examples_prevalent.json
  pathway_examples_gaps.json
  pathway_examples_edgeonly.json
"""

import json
import pickle
import logging
import time
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
PKL_DIR = BASE / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4_DIR = BASE / "phase2_results/step4_finalanalysis"
STEP5_DIR = BASE / "phase2_results/step5_naming"
OUT_DIR = BASE / "phase2_results/step5_examples"
LOG_DIR = BASE.parent / "logfiles/phase5_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase2_step5_examples.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

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

t0 = time.time()
with open(STEP4_DIR / "optionA_kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)
log.info(f"  KMeans model loaded  ({time.time() - t0:.1f}s)")

# ── Build node→cluster dicts ──────────────────────────────────────────────────
log.info("Building valid_pathway_nodes from unconstrained path file …")
PATHS_UNCONSTRAINED = BASE / "phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"
if not PATHS_UNCONSTRAINED.exists():
    PATHS_UNCONSTRAINED = (
        BASE.parent
        / "graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"
    )
# maturity>=3 endpoint filter — path gen used ALL_INTERVENTION_IDS
valid_pathway_nodes = set()
with open(PATHS_UNCONSTRAINED) as _f:
    for _line in _f:
        _obj = json.loads(_line)
        _path = [
            int(x) for x in (_obj.get("path") or _obj.get("node_id_sequence") or [])
        ]
        if not _path:
            continue
        _interv_id = _path[-1]
        if (
            int(node_attrs.get(_interv_id, {}).get("intervention_maturity", 0) or 0)
            >= 3
        ):
            valid_pathway_nodes.update(_path)
log.info(f"  valid_pathway_nodes: {len(valid_pathway_nodes):,} nodes")

log.info("Building node→cluster lookup dicts …")
node_to_risk = {}
node_to_interv = {}
for (ec, mode, nt, algo, cid), members in cluster_memberships.items():
    if (
        ec == 0.9 and mode == "unconstrained" and algo == "agglomerative"
    ):  # ec is float 0.9
        if nt == "risk":
            for nid in members:
                if nid in valid_pathway_nodes:
                    node_to_risk[nid] = int(cid)
        elif nt == "intervention":
            for nid in members:
                if nid in valid_pathway_nodes:
                    node_to_interv[nid] = int(cid)
log.info(f"  node_to_risk: {len(node_to_risk)} nodes")
log.info(f"  node_to_interv: {len(node_to_interv)} nodes")


# ── Load cluster names ────────────────────────────────────────────────────────
def load_names(csv_path, fallback_csv):
    """Load final_name per cluster_id from LLM naming CSV; fall back to algorithmic CSV."""
    names = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names[int(row["cluster_id"])] = row.get("final_name") or row.get(
                    "llm_name", ""
                )
        log.info(f"  Loaded {len(names)} names from {csv_path.name}")
    elif fallback_csv.exists():
        with open(fallback_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names[int(row["cluster_id"])] = row.get("top_node_name") or row.get(
                    "cluster_id", ""
                )
        log.info(f"  Loaded {len(names)} names from fallback {fallback_csv.name}")
    return names


risk_names = load_names(
    STEP5_DIR / "risk_cluster_names_llm.csv",
    STEP4_DIR / "step4_cluster_tables/risk_clusters.csv",
)
interv_names = load_names(
    STEP5_DIR / "intervention_cluster_names_llm.csv",
    STEP4_DIR / "step4_cluster_tables/intervention_clusters.csv",
)
chain_names = load_names(
    STEP5_DIR / "chain_cluster_names_llm.csv",
    STEP4_DIR / "step4_cluster_tables/optionA_chainbody_clusters.csv",
)


# ── Embedding helpers ─────────────────────────────────────────────────────────
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


# ── Load top risk→intervention connections ────────────────────────────────────
log.info("Loading connectivity CSVs …")
ri_edges = []
# Use consim1 connectivity (selected config)
with open(
    STEP4_DIR / "step4_connectivity/risk_to_interv_edges_consim1.csv", newline=""
) as f:
    for row in csv.DictReader(f):
        ri_edges.append(
            (int(row["cluster_a"]), int(row["cluster_b"]), int(row["n_paths"]))
        )
ri_edges.sort(key=lambda x: -x[2])
top15_ri = ri_edges[:15]
log.info(f"  Top 15 risk→intervention: {top15_ri[:3]} …")

# Gap clusters: bottom 10 risk clusters by total paths to any intervention
risk_path_totals = defaultdict(int)
for ra, rb, np_ in ri_edges:
    risk_path_totals[ra] += np_
bottom10_risks = sorted(risk_path_totals, key=lambda x: risk_path_totals[x])[:10]
log.info(f"  Bottom 10 risk clusters (gap): {bottom10_risks}")

# ── Build Option B family lookup ──────────────────────────────────────────────
log.info("Loading Option B co-occurrence families …")
optionB_top10 = []
with open(
    STEP4_DIR / "step4_cluster_tables/optionB_cooccurrence_families.csv", newline=""
) as f:
    rows_b = list(csv.DictReader(f))
# Sort by n_paths descending and take top 10
rows_b.sort(key=lambda r: -int(r["n_paths"]))
optionB_top10 = rows_b[:10]
log.info("  Top 10 Option B families loaded")

# ── Streaming path assignment ──────────────────────────────────────────────────
# Collect path records grouped by (risk_cid, chain_cid, interv_cid)
# We need this for:
#  - prevalent: top-15 risk→interv combos + top-10 chain clusters
#  - gaps: bottom-10 risk clusters

TARGET_RISK_SET = set(rc for rc, _, _ in top15_ri) | set(bottom10_risks)
TARGET_INTERV_SET = set(ic for _, ic, _ in top15_ri)

# Per (risk, chain, interv) triple: list of path records
combo_paths = defaultdict(list)  # key: (r_cid, ch_cid, i_cid)
# Per chain cluster: all paths (for top-10 chain cluster examples)
chain_paths = defaultdict(list)  # key: ch_cid
# Per gap risk cluster: all paths
gap_paths = defaultdict(list)  # key: r_cid

MAX_PER_COMBO = 50  # buffer enough to find EDGE-only + diverse examples
MAX_PER_CHAIN = 20
MAX_PER_GAP = 500  # gap clusters have very few paths total

PATHS_FILE = (
    STEP4_DIR / "step4_paths/representative_pathways_consim1.jsonl"
)  # selected config
PREDICT_BATCH = 2000

log.info("Streaming path file for assignments …")
log.info(
    f"  Targeting {len(TARGET_RISK_SET)} risk clusters, {len(TARGET_INTERV_SET)} interv clusters"
)

batch_embs = []
batch_meta = []  # (path_record, risk_cid, interv_cid)


def flush_batch():
    if not batch_embs:
        return
    labels = kmeans.predict(np.stack(batch_embs))
    for lab, (rec, r_cid, i_cid) in zip(labels, batch_meta):
        ch_cid = int(lab)
        key = (r_cid, ch_cid, i_cid)
        if len(combo_paths[key]) < MAX_PER_COMBO:
            combo_paths[key].append(rec)
        if len(chain_paths[ch_cid]) < MAX_PER_CHAIN:
            chain_paths[ch_cid].append(rec)
        if r_cid in bottom10_risks and len(gap_paths[r_cid]) < MAX_PER_GAP:
            gap_paths[r_cid].append(rec)
    batch_embs.clear()
    batch_meta.clear()


t0 = time.time()
n_processed = 0
with open(PATHS_FILE) as f:
    for line in f:
        obj = json.loads(line)
        path = obj["node_id_sequence"]
        if not path:
            continue
        start, end = path[0], path[-1]
        r_cid = node_to_risk.get(start)
        i_cid = node_to_interv.get(end)
        if r_cid is None or i_cid is None:
            continue
        # Only process if in target set or is a gap cluster
        if r_cid not in TARGET_RISK_SET and r_cid not in bottom10_risks:
            continue

        body_ids = path[1:-1]
        if not body_ids:
            continue
        embs_b = [get_embedding(nid) for nid in body_ids]
        embs_b = [e for e in embs_b if e is not None]
        if not embs_b:
            continue
        mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
        batch_embs.append(mean_emb)
        batch_meta.append((obj, r_cid, i_cid))
        n_processed += 1

        if len(batch_embs) >= PREDICT_BATCH:
            flush_batch()

flush_batch()
log.info(f"  Processed {n_processed:,} paths in {time.time() - t0:.1f}s")
log.info(f"  combo_paths: {len(combo_paths)} distinct (risk, chain, interv) triples")
log.info(f"  chain_paths: {len(chain_paths)} chain clusters with paths")
log.info(f"  gap_paths:   {len(gap_paths)} gap risk clusters with paths")

# ── Also collect EDGE-only paths ─────────────────────────────────────────────
log.info("Loading EDGE-only paths …")
edgeonly_by_combo = defaultdict(list)
edgeonly_by_chain = defaultdict(list)
edgeonly_by_gap = defaultdict(list)

EDGEONLY_FILE = STEP4_DIR / "step4_paths/representative_pathways_edgeonly.jsonl"
eo_batch_embs, eo_batch_meta = [], []


def flush_eo():
    if not eo_batch_embs:
        return
    labels = kmeans.predict(np.stack(eo_batch_embs))
    for lab, (rec, r_cid, i_cid) in zip(labels, eo_batch_meta):
        ch_cid = int(lab)
        key = (r_cid, ch_cid, i_cid)
        edgeonly_by_combo[key].append(rec)
        edgeonly_by_chain[ch_cid].append(rec)
        if r_cid in bottom10_risks:
            edgeonly_by_gap[r_cid].append(rec)
    eo_batch_embs.clear()
    eo_batch_meta.clear()


with open(EDGEONLY_FILE) as f:
    for line in f:
        obj = json.loads(line)
        path = obj["node_id_sequence"]
        if not path:
            continue
        r_cid = node_to_risk.get(path[0])
        i_cid = node_to_interv.get(path[-1])
        if r_cid is None or i_cid is None:
            continue
        body_ids = path[1:-1]
        embs_b = [
            get_embedding(nid) for nid in body_ids if get_embedding(nid) is not None
        ]
        if not embs_b:
            continue
        mean_emb = np.stack(embs_b).mean(axis=0).astype(np.float32)
        eo_batch_embs.append(mean_emb)
        eo_batch_meta.append((obj, r_cid, i_cid))
        if len(eo_batch_embs) >= 500:
            flush_eo()
flush_eo()
log.info(
    f"  EDGE-only: {sum(len(v) for v in edgeonly_by_combo.values())} paths assigned to combos"
)


# ── Example selection helpers ──────────────────────────────────────────────────
def path_to_example(rec, example_type="VarB"):
    """Convert a raw path record to a clean example dict."""
    return {
        "type": example_type,
        "source_url": rec.get("source_url", ""),
        "path_length": rec.get("path_length", len(rec.get("node_id_sequence", []))),
        "max_consec_SIM": rec.get("max_consec_SIM", -1),
        "chain": [
            {"category": cat, "name": name}
            for cat, name in zip(
                rec.get("categories", []),
                rec.get("node_names", []),
            )
        ],
    }


def select_examples(edgeonly_recs, varB_recs, n=3):
    """
    Select up to n examples with priority:
    1. EDGE-only (max_consec_SIM = 0) paths, shortest first
    2. VarB paths with max_consec_SIM = 1, shortest first
    3. VarB paths with max_consec_SIM = 2, shortest first
    Ensure source URL diversity (max 1 per URL).
    """
    seen_urls = set()
    selected = []

    def try_add(rec, etype):
        if len(selected) >= n:
            return
        url = rec.get("source_url", "")
        if url in seen_urls:
            return
        seen_urls.add(url)
        selected.append(path_to_example(rec, etype))

    # Sort each tier by path length ascending
    for rec in sorted(edgeonly_recs, key=lambda r: r.get("path_length", 99)):
        try_add(rec, "EDGE-only")

    for mcs in [1, 2]:
        for rec in sorted(
            [r for r in varB_recs if r.get("max_consec_SIM", -1) == mcs],
            key=lambda r: r.get("path_length", 99),
        ):
            try_add(rec, f"VarB(consec_SIM≤{mcs})")

    return selected


# ── Build prevalent examples ───────────────────────────────────────────────────
log.info("\n── Building prevalent pathway examples ──")

# Top-15 risk→intervention connections
prevalent_ri = []
for r_cid, i_cid, n_paths in top15_ri:
    # Find all combos for this risk→interv pair (different chain clusters)
    all_combos = [
        (r_cid, ch, i_cid) for (r, ch, i) in combo_paths if r == r_cid and i == i_cid
    ]
    # Dominant chain = combo with most paths
    dominant_chain = (
        max(all_combos, key=lambda k: len(combo_paths[k])) if all_combos else None
    )

    examples = []
    if dominant_chain:
        ch_cid = dominant_chain[1]
        examples = select_examples(
            edgeonly_by_combo.get(dominant_chain, []),
            combo_paths[dominant_chain],
            n=3,
        )
    else:
        ch_cid = None

    prevalent_ri.append(
        {
            "connection_type": "risk_to_intervention",
            "risk_cluster": {
                "id": r_cid,
                "name": risk_names.get(r_cid, f"risk_{r_cid}"),
            },
            "intervention_cluster": {
                "id": i_cid,
                "name": interv_names.get(i_cid, f"interv_{i_cid}"),
            },
            "dominant_chain_cluster": {
                "id": ch_cid,
                "name": chain_names.get(ch_cid, f"chain_{ch_cid}")
                if ch_cid is not None
                else "",
            }
            if ch_cid is not None
            else None,
            "n_paths_total": n_paths,
            "n_paths_in_dominant_chain": len(combo_paths.get(dominant_chain, []))
            if dominant_chain
            else 0,
            "examples": examples,
        }
    )
    log.info(
        f"  R{r_cid}→I{i_cid}: {n_paths:,} paths, dominant chain C{ch_cid}, {len(examples)} examples"
    )

# Top-10 Option A chain clusters
prevalent_chains = []
# Sort chain clusters by total paths collected
chain_path_counts = {ch: len(paths) for ch, paths in chain_paths.items()}
top10_chains = sorted(chain_path_counts, key=lambda x: -chain_path_counts[x])[:10]

for ch_cid in top10_chains:
    examples = select_examples(
        edgeonly_by_chain.get(ch_cid, []),
        chain_paths[ch_cid],
        n=3,
    )
    prevalent_chains.append(
        {
            "connection_type": "chain_cluster",
            "chain_cluster": {
                "id": ch_cid,
                "name": chain_names.get(ch_cid, f"chain_{ch_cid}"),
            },
            "n_paths_collected": chain_path_counts[ch_cid],
            "examples": examples,
        }
    )
    log.info(
        f"  Chain cluster {ch_cid}: {chain_path_counts[ch_cid]} paths, {len(examples)} examples"
    )

# Top-10 Option B co-occurrence families
prevalent_optionB = []
for row in optionB_top10:
    fid = int(row["family_id"])
    sig = row.get("signature_str", "")
    n_paths = int(row["n_paths"])
    n_src = int(row["n_sources"])
    prevalent_optionB.append(
        {
            "connection_type": "optionB_family",
            "family_id": fid,
            "n_paths": n_paths,
            "n_sources": n_src,
            "signature": sig,
            "note": (
                "Option B family defined by co-occurring (subtype, cluster_id) signature. "
                "Example pathways are not retrievable without per-path Option B family assignment "
                "(would require re-streaming with subtype cluster lookup). "
                "Use Option A chain cluster examples for workshop paper; Option B for quantitative analysis."
            ),
        }
    )

prevalent_output = {
    "description": (
        "Prevalent pathway examples: top-15 risk→intervention connections, "
        "top-10 Option A chain clusters, top-10 Option B families."
    ),
    "generated": "2026-03-29",
    "top15_risk_to_intervention": prevalent_ri,
    "top10_chain_clusters_optionA": prevalent_chains,
    "top10_families_optionB": prevalent_optionB,
}

with open(OUT_DIR / "pathway_examples_prevalent.json", "w", encoding="utf-8") as f:
    json.dump(prevalent_output, f, indent=2, ensure_ascii=False)
log.info("Saved pathway_examples_prevalent.json")

# ── Build gap examples ────────────────────────────────────────────────────────
log.info("\n── Building gap pathway examples ──")

gap_output_list = []
for r_cid in bottom10_risks:
    all_recs = gap_paths.get(r_cid, [])
    eo_recs = edgeonly_by_gap.get(r_cid, [])
    n_total = risk_path_totals.get(r_cid, 0)
    ri_for_risk = [(ra, rb, np_) for ra, rb, np_ in ri_edges if ra == r_cid]
    n_distinct_interv = len(ri_for_risk)

    # For gap clusters: show ALL available paths (often just 3-8)
    examples = select_examples(
        eo_recs, all_recs, n=min(5, max(len(eo_recs) + len(all_recs), 1))
    )

    gap_output_list.append(
        {
            "risk_cluster": {
                "id": r_cid,
                "name": risk_names.get(r_cid, f"risk_{r_cid}"),
            },
            "gap_metrics": {
                "total_paths_to_any_intervention": n_total,
                "n_distinct_intervention_clusters": n_distinct_interv,
                "has_edge_only_path": len(eo_recs) > 0,
                "interpretation": (
                    "Low path count indicates limited intervention literature for this risk category "
                    "in the ARD corpus. May reflect corpus coverage limits rather than genuine "
                    "research absence — verify against external literature before concluding this "
                    "is an open research gap."
                ),
            },
            "connected_interventions": [
                {
                    "intervention_cluster_id": rb,
                    "intervention_name": interv_names.get(rb, f"interv_{rb}"),
                    "n_paths": np_,
                }
                for ra, rb, np_ in sorted(ri_for_risk, key=lambda x: -x[2])
            ],
            "examples": examples,
        }
    )
    log.info(
        f"  Gap R{r_cid} '{risk_names.get(r_cid, '')[:50]}': "
        f"{n_total} total paths, {n_distinct_interv} interventions, "
        f"{len(eo_recs)} EDGE-only, {len(examples)} examples"
    )

gap_output = {
    "description": (
        "Gap pathway examples: bottom-10 risk clusters by total paths to interventions. "
        "These represent risk areas with the thinnest intervention literature in the ARD corpus."
    ),
    "generated": "2026-03-29",
    "gap_clusters": gap_output_list,
}

with open(OUT_DIR / "pathway_examples_gaps.json", "w", encoding="utf-8") as f:
    json.dump(gap_output, f, indent=2, ensure_ascii=False)
log.info("Saved pathway_examples_gaps.json")

# ── Build EDGE-only examples (strongest single-paper evidence) ────────────────
log.info("\n── Building EDGE-only examples ──")
edgeonly_output_list = []
# Group all EDGE-only paths by (risk, interv) pair and pick best 2 per pair
eo_by_ri = defaultdict(list)
for (r, ch, i), recs in edgeonly_by_combo.items():
    eo_by_ri[(r, i)].extend(recs)

# Top 20 risk→interv pairs by n_edge_only paths
top_eo = sorted(eo_by_ri.items(), key=lambda x: -len(x[1]))[:20]
for (r_cid, i_cid), recs in top_eo:
    examples = select_examples(recs, [], n=2)
    edgeonly_output_list.append(
        {
            "risk_cluster": {
                "id": r_cid,
                "name": risk_names.get(r_cid, f"risk_{r_cid}"),
            },
            "intervention_cluster": {
                "id": i_cid,
                "name": interv_names.get(i_cid, f"interv_{i_cid}"),
            },
            "n_edge_only_paths": len(recs),
            "examples": examples,
        }
    )

edgeonly_output = {
    "description": (
        "EDGE-only pathway examples: top-20 risk→intervention pairs by number of EDGE-only paths. "
        "These paths have max_consec_SIM=0 — every edge is a structural EDGE (conf≥3) from a single paper, "
        "representing the strongest single-source causal chain evidence in the corpus."
    ),
    "generated": "2026-03-29",
    "top20_by_edge_only_count": edgeonly_output_list,
}

with open(OUT_DIR / "pathway_examples_edgeonly.json", "w", encoding="utf-8") as f:
    json.dump(edgeonly_output, f, indent=2, ensure_ascii=False)
log.info("Saved pathway_examples_edgeonly.json")

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("\n── Summary ──")
log.info(
    f"  Prevalent: {len(prevalent_ri)} risk→interv, {len(prevalent_chains)} chain clusters, {len(prevalent_optionB)} option B families"
)
log.info(f"  Gaps: {len(gap_output_list)} risk clusters")
log.info(f"  EDGE-only: {len(edgeonly_output_list)} risk→interv pairs")
log.info("Done.")
