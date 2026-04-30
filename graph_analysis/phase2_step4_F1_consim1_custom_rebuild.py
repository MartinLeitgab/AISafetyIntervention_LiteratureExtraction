"""
phase2_step4_F1_consim1_custom_rebuild.py [rev8]

Rebuilds the consim1 analysis universe on custom-mode paths.

Pipeline (single pass on paths_custom_sim0.9.jsonl):
  1. Build vpn_custom (nodes on custom paths with maturity>=3 endpoint).
  2. Build sim_edge_set (SIM>=0.9 within vpn_custom).
  3. Apply consim1 filter (max_consec_sim<=1) to get qualifying paths.
  4. Build node_to_stc (body-subtype mappings, restricted to vpn_custom).
  5. Compute frozenset signature per qualifying path.
  6. Aggregate (sig_counts, source diversity, top subtypes) -> family CSV.
  7. Aggregate (risk_cluster, family, interv_cluster) -> ri_triplets CSV.

Outputs (all suffixed _custom_consim1 to preserve rev7 outputs):
  step4_paths/representative_pathways_custom_consim1.jsonl
  step4_cluster_tables/optionB_cooccurrence_families_custom_consim1.csv
  step4_connectivity/ri_triplets_custom_consim1.csv
  step4_connectivity/custom_consim1_summary.txt

Inputs:
  phase1_rawpathsfiles/paths_custom_sim0.9.jsonl
  step1_load_and_parse.../cluster_memberships.pkl
  step1_load_and_parse.../graph_node_attributes.pkl
  step1_load_and_parse.../graph_edge_data.pkl
  step5_naming/risk_cluster_names_llm_v2.csv  (names for ri_triplets)
  step5_naming/intervention_cluster_names_llm_v2.csv
"""

import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
PATHS_DIR = ROOT / "phase1_rawpathsfiles"
STEP1_DIR = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
STEP4_DIR = ROOT / "phase2_results/step4_finalanalysis"
NAMING_DIR = ROOT / "phase2_results/step5_naming"
OUT_PATHS = STEP4_DIR / "step4_paths"
OUT_TABLES = STEP4_DIR / "step4_cluster_tables"
OUT_CONN = STEP4_DIR / "step4_connectivity"
for d in [OUT_PATHS, OUT_TABLES, OUT_CONN]:
    d.mkdir(parents=True, exist_ok=True)

CUSTOM_PATHS = PATHS_DIR / "paths_custom_sim0.9.jsonl"

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


print("=" * 70)
print("Phase 2 Step 4 rev8 — consim1 rebuild on custom paths")
print("=" * 70)

# ─── Load inputs ──────────────────────────────────────────────────────────────
t0 = time.time()
print("Loading PKL files...")
with open(STEP1_DIR / "cluster_memberships.pkl", "rb") as f:
    cm = pickle.load(f)
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)
print(f"  Loaded in {time.time() - t0:.1f}s")

# ─── Pass 1: build vpn_custom and quality-filter custom paths ─────────────────
print(f"\nReading {CUSTOM_PATHS.name} ...")
t1 = time.time()
vpn_custom = set()
n_custom_total = 0
n_custom_mature = 0
custom_obj_keep = []
with open(CUSTOM_PATHS) as f:
    for line in f:
        obj = json.loads(line)
        path = [int(x) for x in obj["path"]]
        n_custom_total += 1
        interv_id = path[-1]
        if int(node_attrs.get(interv_id, {}).get("intervention_maturity", 0) or 0) >= 3:
            n_custom_mature += 1
            vpn_custom.update(path)
            custom_obj_keep.append(obj)

print(
    f"  {n_custom_total:,} custom paths total | {n_custom_mature:,} with maturity>=3 endpoint"
)
print(f"  vpn_custom: {len(vpn_custom):,} nodes  ({time.time() - t1:.1f}s)")

# ─── Build sim_edge_set (SIM>=0.9 within vpn_custom) ──────────────────────────
print("\nBuilding sim_edge_set (SIM>=0.9, vpn_custom-restricted)...")
t2 = time.time()
sim_edge_set = set()
for e in edge_data:
    if str(e.get("type", "")).upper() != "SIMILARITY":
        continue
    score = e.get("similarity_score")
    if score is None or cos_sim_from_score(score) < 0.9:
        continue
    try:
        s, t = int(e["source"]), int(e["target"])
        if s in vpn_custom and t in vpn_custom:
            sim_edge_set.add((min(s, t), max(s, t)))
    except (ValueError, TypeError):
        pass
print(f"  {len(sim_edge_set):,} sim_edge_set pairs  ({time.time() - t2:.1f}s)")


def max_consec_sim(path):
    max_run = run = 0
    for i in range(len(path) - 1):
        a, b = int(path[i]), int(path[i + 1])
        if (min(a, b), max(a, b)) in sim_edge_set:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


# ─── Apply consim1 filter (max_consec_sim<=1) ─────────────────────────────────
print("\nApplying consim1 filter (max_consec_sim<=1)...")
t3 = time.time()
qual_paths = []
for obj in custom_obj_keep:
    path = [int(x) for x in obj["path"]]
    if max_consec_sim(path) <= 1:
        qual_paths.append({"path": path, "categories": obj.get("categories", [])})
n_qual = len(qual_paths)
print(
    f"  qualifying paths (custom + maturity>=3 + consim1): {n_qual:,}  ({time.time() - t3:.1f}s)"
)

# Write qualifying path file (compatible with rev7 representative_pathways_consim1 format)
out_qual = OUT_PATHS / "representative_pathways_custom_consim1.jsonl"
with open(out_qual, "w") as f:
    for p in qual_paths:
        f.write(
            json.dumps({"node_id_sequence": p["path"], "categories": p["categories"]})
            + "\n"
        )
print(f"  Written: {out_qual.name}")

# ─── Build node_to_stc (body subtype mappings) ────────────────────────────────
print("\nBuilding node_to_stc...")
node_to_stc = {}
for (ec, mode, nt, algo, cid), members in cm.items():
    try:
        ec_float = float(ec)
    except Exception:
        continue
    if (
        abs(ec_float - 0.9) < 1e-9
        and str(mode) == "unconstrained"
        and str(nt) in BODY_SUBTYPES
        and str(algo) == "agglomerative"
    ):
        for nid in members:
            node_to_stc[int(nid)] = (str(nt), str(cid))
print(f"  node_to_stc: {len(node_to_stc):,} body nodes mapped")


# ─── Build risk and intervention clusters (filtered to vpn_custom) ────────────
def get_clusters(node_type):
    result = {}
    for (ec, mode, nt, algo, cid), members in cm.items():
        try:
            ec_float = float(ec)
        except Exception:
            continue
        if (
            abs(ec_float - 0.9) < 1e-9
            and str(mode) == "unconstrained"
            and str(nt) == node_type
            and str(algo) == "agglomerative"
        ):
            result[str(cid)] = [int(n) for n in members]
    return result


risk_clusters_base = get_clusters("risk")
interv_clusters_base = get_clusters("intervention")

risk_clusters = {
    cid: [n for n in nodes if n in vpn_custom]
    for cid, nodes in risk_clusters_base.items()
}
risk_clusters = {cid: nodes for cid, nodes in risk_clusters.items() if nodes}

interv_clusters = {
    cid: [
        n
        for n in nodes
        if n in vpn_custom
        and int(node_attrs.get(n, {}).get("intervention_maturity", 0) or 0) >= 3
    ]
    for cid, nodes in interv_clusters_base.items()
}
interv_clusters = {cid: nodes for cid, nodes in interv_clusters.items() if nodes}

node_to_risk = {nid: cid for cid, nodes in risk_clusters.items() for nid in nodes}
node_to_interv = {nid: cid for cid, nodes in interv_clusters.items() for nid in nodes}
print(
    f"  risk_clusters: {len(risk_clusters)} (post-VPN), interv_clusters: {len(interv_clusters)}"
)

# ─── Build frozenset signatures and aggregate ─────────────────────────────────
print("\nBuilding frozenset signatures...")
t4 = time.time()
sig_counts = Counter()
sig_sources = defaultdict(set)
sig_top_subtype_counts = defaultdict(Counter)

# For ri_triplets: per-path (risk_cid, sig, interv_cid) tracking
triplet_counts = Counter()  # (risk_cid, sig, interv_cid) -> n
r2f_counts = Counter()  # (risk_cid, sig) -> n
f2i_counts = Counter()  # (sig, interv_cid) -> n

for p in qual_paths:
    path = p["path"]
    if len(path) < 3:
        continue
    body = path[1:-1]
    sig = frozenset(node_to_stc[n] for n in body if n in node_to_stc)
    if not sig:
        continue
    sig_counts[sig] += 1

    # Track per-path source URL for n_sources
    risk_node = path[0]
    interv_node = path[-1]
    # Source URL is on each node; use the source paper of intervention node
    src_url = str(node_attrs.get(interv_node, {}).get("url", ""))
    if src_url and src_url not in {"", "None", "nan"}:
        sig_sources[sig].add(src_url)

    # Top subtype tracking (count subtypes appearing in body)
    for nid in body:
        attrs = node_attrs.get(nid, {})
        cat = str(attrs.get("concept_category", "")).strip()
        if cat in BODY_SUBTYPES:
            sig_top_subtype_counts[sig][cat] += 1

    # Triplet aggregation
    risk_cid = node_to_risk.get(risk_node)
    interv_cid = node_to_interv.get(interv_node)
    if risk_cid and interv_cid:
        triplet_counts[(risk_cid, sig, interv_cid)] += 1
        r2f_counts[(risk_cid, sig)] += 1
        f2i_counts[(sig, interv_cid)] += 1

print(f"  Total unique frozensets: {len(sig_counts):,} ({time.time() - t4:.1f}s)")

# Filter to n>=5
large_sigs = sorted(
    [s for s, c in sig_counts.items() if c >= 5],
    key=lambda s: -sig_counts[s],
)
print(f"  Frozensets with n_paths>=5: {len(large_sigs):,}")

# Assign family_ids in descending n_paths order
sig_to_fam = {sig: i for i, sig in enumerate(large_sigs)}


def sig_to_str(sig):
    parts = []
    for st, cid in sig:
        prefix = SUBTYPE_PREFIX.get(st, st[:2])
        parts.append(f"{prefix}:{cid}")
    return " & ".join(sorted(parts))


# ─── Output: optionB_cooccurrence_families_custom_consim1.csv ─────────────────
print("\nWriting frozenset family CSV...")
fam_rows = []
for sig in large_sigs:
    fam_id = sig_to_fam[sig]
    fam_rows.append(
        {
            "family_id": fam_id,
            "n_paths": sig_counts[sig],
            "n_sources": len(sig_sources[sig]),
            "signature_str": sig_to_str(sig),
            "top_subtypes": str(dict(sig_top_subtype_counts[sig].most_common(5))),
        }
    )
fam_df = pd.DataFrame(fam_rows)
fam_path = OUT_TABLES / "optionB_cooccurrence_families_custom_consim1.csv"
fam_df.to_csv(fam_path, index=False)
print(f"  Written: {fam_path.name} ({len(fam_df)} rows)")

# ─── Output: ri_triplets_custom_consim1.csv ───────────────────────────────────
print("\nWriting ri_triplets CSV...")
# Load name maps for risk/intervention
risk_df = pd.read_csv(NAMING_DIR / "risk_cluster_names_llm_v2.csv")
interv_df = pd.read_csv(NAMING_DIR / "intervention_cluster_names_llm_v2.csv")
risk_name_col = "final_name" if "final_name" in risk_df.columns else "llm_name"
interv_name_col = "final_name" if "final_name" in interv_df.columns else "llm_name"
risk_name_map = dict(
    zip(risk_df["cluster_id"].astype(str), risk_df[risk_name_col].astype(str))
)
interv_name_map = dict(
    zip(
        interv_df["cluster_id"].astype(str),
        interv_df[interv_name_col].astype(str),
    )
)

trip_rows = []
for (risk_cid, sig, interv_cid), n_trip in triplet_counts.items():
    if sig not in sig_to_fam:
        # Family below n_paths>=5 cutoff; skip
        continue
    fam_id = sig_to_fam[sig]
    n_r2c = r2f_counts[(risk_cid, sig)]
    n_c2i = f2i_counts[(sig, interv_cid)]
    trip_rows.append(
        {
            "risk_cid": int(risk_cid),
            "bfamily_id": fam_id,
            "n_paths_r2c": n_r2c,
            "interv_cid": int(interv_cid),
            "n_paths_c2i": n_c2i,
            "n_triplet_paths": n_trip,
            "risk_name": risk_name_map.get(risk_cid, "?"),
            "chain_name": "",  # to be filled by LLM naming step
            "interv_name": interv_name_map.get(interv_cid, "?"),
        }
    )
trip_df = pd.DataFrame(trip_rows).sort_values("n_triplet_paths", ascending=False)
trip_path = OUT_CONN / "ri_triplets_custom_consim1.csv"
trip_df.to_csv(trip_path, index=False)
print(f"  Written: {trip_path.name} ({len(trip_df)} rows)")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY (rev8 custom_consim1)")
print("=" * 70)
print(f"  Custom paths total:                     {n_custom_total:,}")
print(f"  Custom paths with maturity>=3 endpoint: {n_custom_mature:,}")
print(f"  Qualifying paths (consim1 filter):      {n_qual:,}")
print(f"  vpn_custom nodes:                       {len(vpn_custom):,}")
print(f"  sim_edge_set pairs:                     {len(sim_edge_set):,}")
print(f"  Risk clusters (post-VPN):               {len(risk_clusters)}")
print(f"  Intervention clusters (post-VPN):       {len(interv_clusters)}")
print(
    f"  Total unique frozensets:                {len(sig_counts):,} (n>=5: {len(large_sigs):,})"
)
print(f"  ri_triplets rows:                       {len(trip_df):,}")
print("  Top-3 R->I triplets by n_triplet_paths:")
for _, r in trip_df.head(3).iterrows():
    print(
        f"    [{r['risk_name'][:40]}] -> [{r['interv_name'][:40]}]  ({r['n_triplet_paths']} paths)"
    )

# Write summary file
with open(OUT_CONN / "custom_consim1_summary.txt", "w") as f:
    f.write("rev8 custom_consim1 rebuild summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"custom_paths_total: {n_custom_total}\n")
    f.write(f"custom_paths_mature: {n_custom_mature}\n")
    f.write(f"qualifying_paths_consim1: {n_qual}\n")
    f.write(f"vpn_nodes: {len(vpn_custom)}\n")
    f.write(f"sim_edges: {len(sim_edge_set)}\n")
    f.write(f"risk_clusters: {len(risk_clusters)}\n")
    f.write(f"interv_clusters: {len(interv_clusters)}\n")
    f.write(f"unique_frozensets: {len(sig_counts)}\n")
    f.write(f"frozensets_n_ge_5: {len(large_sigs)}\n")
    f.write(f"ri_triplet_rows: {len(trip_df)}\n")
print("\nDONE.")
