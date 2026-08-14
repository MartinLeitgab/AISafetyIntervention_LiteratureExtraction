"""
Phase 2 Step 5 — EDGE-only pathway examples fix + Option B family path examples

Reads directly from raw phase1 path files (not step4 representative files),
builds full node chains with name/description from node_attrs, and also generates
Option B family path examples by matching paths to their family signature.

Outputs (step5_examples/):
  pathway_examples_edgeonly.json    -- top-20 EDGE-only pairs, rebuilt from raw paths
  pathway_examples_optionB.json     -- top-10 Option B families with example paths
"""

import csv
import json
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
PKL_DIR = BASE / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
RAW_PATHS_DIR = BASE / "phase1_rawpathsfiles"
STEP4_DIR = BASE / "phase2_results/step4_finalanalysis"
STEP5_DIR = BASE / "phase2_results/step5_naming"
OUT_DIR = BASE / "phase2_results/step5_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Load PKLs ─────────────────────────────────────────────────────────────────
log.info("Loading cluster_memberships.pkl …")
t0 = time.time()
with open(PKL_DIR / "cluster_memberships.pkl", "rb") as f:
    cluster_memberships = pickle.load(f)
log.info(f"  {len(cluster_memberships)} keys  ({time.time() - t0:.1f}s)")

log.info("Loading graph_node_attributes.pkl …")
t0 = time.time()
with open(PKL_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
log.info(f"  {len(node_attrs)} nodes  ({time.time() - t0:.1f}s)")

# ── Build valid_pathway_nodes ─────────────────────────────────────────────────
log.info("Building valid_pathway_nodes from paths_unconstrained_sim0.9.jsonl …")
t0 = time.time()
valid_pathway_nodes: set = set()
sim09_file = RAW_PATHS_DIR / "paths_unconstrained_sim0.9.jsonl"
with open(sim09_file) as f:
    for line in f:
        obj = json.loads(line)
        path = obj.get("path") or obj.get("node_id_sequence") or []
        for nid in path:
            valid_pathway_nodes.add(int(nid))
log.info(
    f"  {len(valid_pathway_nodes):,} valid_pathway_nodes  ({time.time() - t0:.1f}s)"
)

# ── Build node→cluster dicts (ec=0.9, unconstrained, agglomerative) ───────────
log.info("Building node_to_risk and node_to_interv dicts …")
node_to_risk: dict = {}
node_to_interv: dict = {}
for (ec, mode, nt, algo, cid), members in cluster_memberships.items():
    if ec == 0.9 and mode == "unconstrained" and algo == "agglomerative":
        if nt == "risk":
            for nid in members:
                if nid in valid_pathway_nodes:
                    node_to_risk[nid] = int(cid)
        elif nt == "intervention":
            for nid in members:
                if nid in valid_pathway_nodes:
                    node_to_interv[nid] = int(cid)
log.info(f"  node_to_risk: {len(node_to_risk):,} nodes")
log.info(f"  node_to_interv: {len(node_to_interv):,} nodes")


# ── Load cluster names ────────────────────────────────────────────────────────
def load_names(csv_path: Path, fallback_csv: Path) -> dict:
    """Load final_name per cluster_id from LLM naming CSV, fall back to algorithmic."""
    names: dict = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = int(row["cluster_id"])
                names[cid] = row.get("final_name") or row.get("llm_name") or ""
        log.info(f"  {len(names)} names from {csv_path.name}")
    elif fallback_csv.exists():
        with open(fallback_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = int(row["cluster_id"])
                names[cid] = row.get("top_node_name") or str(cid)
        log.info(f"  {len(names)} names from fallback {fallback_csv.name}")
    return names


risk_names = load_names(
    STEP5_DIR / "risk_cluster_names_llm.csv",
    STEP4_DIR / "step4_cluster_tables/risk_clusters.csv",
)
interv_names = load_names(
    STEP5_DIR / "intervention_cluster_names_llm.csv",
    STEP4_DIR / "step4_cluster_tables/intervention_clusters.csv",
)


# ── Helper: build node chain from path ───────────────────────────────────────
def build_chain(path: list) -> list:
    """Build list of {node_id, category, name, description} for each node in path."""
    chain = []
    for nid in path:
        attrs = node_attrs.get(int(nid), {})
        # category: concept_category for concepts, 'intervention' for interventions
        node_type = str(attrs.get("type", "")).lower()
        if node_type == "intervention":
            category = "intervention"
        else:
            category = str(attrs.get("concept_category") or "concept")
            if category in ("None", "nan", ""):
                category = "concept"
        chain.append(
            {
                "node_id": int(nid),
                "category": category,
                "name": str(attrs.get("name") or ""),
                "description": str(attrs.get("description") or ""),
            }
        )
    return chain


# ── SECTION 1: Top-20 EDGE-only pairs ─────────────────────────────────────────
log.info("=" * 60)
log.info("SECTION 1: Building top-20 EDGE-only pair examples")
log.info("  Streaming paths_unconstrained_edge_only.jsonl …")

edgeonly_file = RAW_PATHS_DIR / "paths_unconstrained_edge_only.jsonl"

# For each (risk_cid, interv_cid): total count + up to 50 candidate records
eo_counts: dict = defaultdict(int)
eo_candidates: dict = defaultdict(list)  # (risk_cid, interv_cid) → list of path dicts
MAX_CANDIDATES = 50

t0 = time.time()
n_total_eo = 0
n_skipped_eo = 0

with open(edgeonly_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # path key can be 'path' or 'node_id_sequence'
        path = obj.get("path") or obj.get("node_id_sequence") or []
        if not path or len(path) < 2:
            continue

        start = int(path[0])
        end = int(path[-1])
        risk_cid = node_to_risk.get(start)
        interv_cid = node_to_interv.get(end)

        if risk_cid is None or interv_cid is None:
            n_skipped_eo += 1
            continue

        n_total_eo += 1
        key = (risk_cid, interv_cid)
        eo_counts[key] += 1

        if len(eo_candidates[key]) < MAX_CANDIDATES:
            # Get source URL from start node (risk node)
            source_url = str(node_attrs.get(start, {}).get("url") or "")
            if not source_url or source_url in ("None", "nan"):
                source_url = str(node_attrs.get(end, {}).get("url") or "")
            eo_candidates[key].append(
                {
                    "path": [int(x) for x in path],
                    "path_length": len(path) - 1,  # n edges
                    "source_url": source_url,
                }
            )

log.info(f"  Processed {n_total_eo:,} EDGE-only paths  ({time.time() - t0:.1f}s)")
log.info(f"  Skipped {n_skipped_eo:,} paths (no cluster assignment)")
log.info(f"  {len(eo_counts):,} unique (risk, interv) pairs")

# Sort pairs by count descending, take top-20
top20_pairs = sorted(eo_counts.items(), key=lambda x: -x[1])[:20]

log.info("Top-20 EDGE-only pairs:")
for (r, i), cnt in top20_pairs:
    log.info(
        f"  R{r} ({risk_names.get(r, '')[:40]}) → I{i} ({interv_names.get(i, '')[:40]}): {cnt} paths"
    )


def select_eo_examples(candidates: list, n: int = 3) -> list:
    """Select up to n examples: prefer shortest paths, diverse source URLs."""
    seen_urls: set = set()
    selected = []
    # Sort by path_length ascending
    for rec in sorted(candidates, key=lambda r: r["path_length"]):
        if len(selected) >= n:
            break
        url = rec["source_url"]
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        selected.append(rec)
    return selected


edgeonly_output_list = []
for (r_cid, i_cid), n_paths in top20_pairs:
    candidates = eo_candidates[(r_cid, i_cid)]
    best_examples = select_eo_examples(candidates, n=3)

    examples = []
    for rec in best_examples:
        examples.append(
            {
                "type": "EDGE-only",
                "source_url": rec["source_url"],
                "path_length": rec["path_length"],
                "chain": build_chain(rec["path"]),
            }
        )

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
            "n_edge_only_paths": n_paths,
            "examples": examples,
        }
    )

edgeonly_output = {
    "description": (
        "Top-20 risk\u2192intervention pairs by EDGE-only path count. "
        "All paths have max_consec_SIM=0 (every edge is structural EDGE conf\u22653 "
        "from a single paper). Built directly from paths_unconstrained_edge_only.jsonl."
    ),
    "generated": "2026-04-05",
    "top20_by_edge_only_count": edgeonly_output_list,
}

out_eo = OUT_DIR / "pathway_examples_edgeonly.json"
with open(out_eo, "w", encoding="utf-8") as f:
    json.dump(edgeonly_output, f, indent=2, ensure_ascii=False)
log.info(f"Saved {out_eo}")


# ── SECTION 2: Option B family path examples ──────────────────────────────────
log.info("=" * 60)
log.info("SECTION 2: Building Option B family path examples")

# Build node_to_stc: body subtype → (subtype, cluster_id) mapping
BODY_SUBTYPES = [
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]

log.info("  Building node_to_stc (body subtype cluster membership) …")
t0 = time.time()
node_to_stc: dict = {}
for (ec, mode, nt, algo, cid), members in cluster_memberships.items():
    if (
        ec == 0.9
        and mode == "unconstrained"
        and algo == "agglomerative"
        and nt in BODY_SUBTYPES
    ):
        for nid in members:
            if nid in valid_pathway_nodes:
                node_to_stc[nid] = (nt, int(cid))
log.info(
    f"  {len(node_to_stc):,} nodes mapped to subtype clusters  ({time.time() - t0:.1f}s)"
)

# Load top-10 Option B families from CSV
log.info("  Loading optionB_cooccurrence_families.csv …")
optionB_rows = []
with open(
    STEP4_DIR / "step4_cluster_tables/optionB_cooccurrence_families.csv", newline=""
) as f:
    for row in csv.DictReader(f):
        optionB_rows.append(row)
optionB_rows.sort(key=lambda r: -int(r["n_paths"]))
top10_families = optionB_rows[:10]
log.info(f"  Loaded {len(optionB_rows)} families, using top 10")

# Parse signature_str back to frozenset of (subtype, cluster_id)
# Format: "de:15 & im:4 & pr:6 & th:11 & va:10"
# Subtype prefix mapping: first 2 chars of subtype name
SUBTYPE_PREFIX: dict = {st[:2]: st for st in BODY_SUBTYPES}


def parse_sig_str(sig_str: str) -> frozenset:
    """Parse 'de:15 & im:4' → frozenset({('design_rationale', 15), ('implementation_mechanism', 4)})"""
    parts = [p.strip() for p in sig_str.split("&")]
    result = set()
    for part in parts:
        if ":" not in part:
            continue
        prefix, cid_str = part.split(":", 1)
        prefix = prefix.strip()
        cid = int(cid_str.strip())
        subtype = SUBTYPE_PREFIX.get(prefix)
        if subtype:
            result.add((subtype, cid))
    return frozenset(result)


# Build family_id → (frozenset_sig, row) mapping for top-10
fid_to_sig: dict = {}
for row in top10_families:
    fid = int(row["family_id"])
    sig_str = row["signature_str"]
    sig = parse_sig_str(sig_str)
    fid_to_sig[fid] = sig

# Build reverse: frozenset_sig → family_id (for lookup during streaming)
sig_to_fid: dict = {sig: fid for fid, sig in fid_to_sig.items()}
target_sigs: set = set(sig_to_fid.keys())

# Stream paths_unconstrained_sim0.9.jsonl and match to families
# Use consim1 representative paths as they have source_url and names
# But we need to also handle the raw sim09 paths if needed
# Use the step4 representative_pathways_consim1.jsonl (has node_names, source_url)
consim1_file = STEP4_DIR / "step4_paths/representative_pathways_consim1.jsonl"

log.info(f"  Streaming {consim1_file.name} for Option B family assignment …")
t0 = time.time()

# Per family_id: up to 50 candidate path records
fam_candidates: dict = defaultdict(list)
MAX_FAM_CANDIDATES = 50
n_fam_matched = 0
n_fam_processed = 0

with open(consim1_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        path_seq = obj.get("node_id_sequence") or obj.get("path") or []
        if not path_seq or len(path_seq) < 3:
            continue

        # Convert path to ints
        path_ints = [int(x) for x in path_seq]
        body = path_ints[1:-1]

        # Compute signature
        sig_parts = frozenset(node_to_stc[n] for n in body if n in node_to_stc)
        if not sig_parts or sig_parts not in target_sigs:
            n_fam_processed += 1
            continue

        fid = sig_to_fid[sig_parts]
        n_fam_matched += 1
        n_fam_processed += 1

        if len(fam_candidates[fid]) < MAX_FAM_CANDIDATES:
            source_url = str(obj.get("source_url") or "")
            max_csim = obj.get("max_consec_SIM", -1)
            try:
                max_csim = int(max_csim)
            except (ValueError, TypeError):
                max_csim = -1

            # Build chain using node_names from consim1 + description from node_attrs
            node_names = obj.get("node_names", [])
            categories = obj.get("categories", [])
            chain = []
            for idx, nid in enumerate(path_ints):
                attrs = node_attrs.get(nid, {})
                name = (
                    node_names[idx]
                    if idx < len(node_names)
                    else str(attrs.get("name") or "")
                )
                category = categories[idx] if idx < len(categories) else ""
                if not category:
                    node_type = str(attrs.get("type", "")).lower()
                    if node_type == "intervention":
                        category = "intervention"
                    else:
                        category = str(attrs.get("concept_category") or "concept")
                chain.append(
                    {
                        "node_id": nid,
                        "category": category,
                        "name": str(name),
                        "description": str(attrs.get("description") or ""),
                    }
                )

            fam_candidates[fid].append(
                {
                    "source_url": source_url,
                    "path_length": len(path_ints) - 1,
                    "max_consec_SIM": max_csim,
                    "chain": chain,
                }
            )

log.info(
    f"  Processed {n_fam_processed:,} paths, matched {n_fam_matched:,} to top-10 families  ({time.time() - t0:.1f}s)"
)
log.info(f"  Families with examples: {len(fam_candidates)}")


def select_fam_examples(candidates: list, n: int = 3) -> list:
    """Select up to n examples: prefer EDGE-only (max_consec_SIM=0), then diverse URLs, shortest."""
    # Sort: EDGE-only first, then by path_length
    sorted_cands = sorted(
        candidates,
        key=lambda r: (
            r["max_consec_SIM"] if r["max_consec_SIM"] >= 0 else 99,
            r["path_length"],
        ),
    )
    seen_urls: set = set()
    selected = []
    for rec in sorted_cands:
        if len(selected) >= n:
            break
        url = rec["source_url"]
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        selected.append(
            {
                "type": "EDGE-only"
                if rec["max_consec_SIM"] == 0
                else f"VarB(consec_SIM\u2264{rec['max_consec_SIM']})",
                "source_url": rec["source_url"],
                "path_length": rec["path_length"],
                "max_consec_SIM": rec["max_consec_SIM"],
                "chain": rec["chain"],
            }
        )
    return selected


optionB_output_list = []
for row in top10_families:
    fid = int(row["family_id"])
    sig_str = row["signature_str"]
    n_paths = int(row["n_paths"])
    n_sources = int(row["n_sources"])
    candidates = fam_candidates.get(fid, [])
    examples = select_fam_examples(candidates, n=3)

    optionB_output_list.append(
        {
            "family_id": fid,
            "n_paths": n_paths,
            "n_sources": n_sources,
            "signature": sig_str,
            "note": (
                "Option B family defined by co-occurring (subtype, cluster_id) frozenset signature "
                "of body nodes. Paths matched by recomputing signature from consim1 representative paths."
            ),
            "examples": examples,
        }
    )
    log.info(
        f"  Family {fid}: {n_paths:,} total paths, {len(examples)} examples, sig={sig_str[:60]}"
    )

optionB_output = {
    "description": (
        "Option B co-occurrence family path examples: top-10 families by n_paths. "
        "Each family is a frozenset of (subtype, cluster_id) co-occurrence signatures "
        "in path body nodes. Examples drawn from consim1 representative paths."
    ),
    "generated": "2026-04-05",
    "top10_families_optionB": optionB_output_list,
}

out_b = OUT_DIR / "pathway_examples_optionB.json"
with open(out_b, "w", encoding="utf-8") as f:
    json.dump(optionB_output, f, indent=2, ensure_ascii=False)
log.info(f"Saved {out_b}")

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("SUMMARY")
log.info(f"  Total EDGE-only paths processed: {n_total_eo:,}")
log.info(f"  Unique (risk, interv) pairs found: {len(eo_counts):,}")
log.info("  Top-20 pairs:")
for (r, i), cnt in top20_pairs:
    log.info(
        f"    R{r}({risk_names.get(r, '')[:35]}) → I{i}({interv_names.get(i, '')[:35]}): {cnt}"
    )
log.info(f"  Output: {out_eo}")
log.info(
    f"  Option B families with examples: {len(fam_candidates)}/{len(top10_families)}"
)
log.info(f"  Output: {out_b}")
log.info("Done.")
