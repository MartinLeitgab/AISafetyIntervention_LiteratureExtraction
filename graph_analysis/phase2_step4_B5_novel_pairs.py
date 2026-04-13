"""
Phase 2 Track B5 — Top-20 Novel SIM-bridged-only R→I pairs with novelty filter
================================================================================
Revision plan item 10.

From cross_config_ri_pairs.csv (1,289 R→I cluster pairs), filter to pairs where
n_paths_c0 = 0 (no EDGE-only path = never argued end-to-end in a single paper).
These 685 pairs are established ONLY via cross-paper semantic (SIM) bridging.

Apply novelty filter:
  - Exclude pairs where intervention is a generic "field-building" intervention
    (fund AI safety research, education/outreach) — these are trivially motivated
    by almost any risk and therefore not novel insights
  - Keep technical and governance interventions as more specific and novel

Output: top-20 filtered novel SIM-bridged-only R→I pairs with risk/intervention names

Note: web search for post-2023 papers is NOT automated here — a manual check table
is provided that a human reviewer can fill in.
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "phase2_results")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4_finalanalysis")
CONN_DIR = os.path.join(STEP4_DIR, "step4_connectivity")
TABLES_DIR = os.path.join(STEP4_DIR, "step4_cluster_tables")

# ─── Load data ─────────────��─────────────────────────────────────────────────
ri_pairs = pd.read_csv(os.path.join(CONN_DIR, "cross_config_ri_pairs.csv"))
risk_names = pd.read_csv(os.path.join(TABLES_DIR, "risk_cluster_names.csv"))
interv_names = pd.read_csv(os.path.join(TABLES_DIR, "intervention_cluster_names.csv"))

print(f"Total R→I cluster pairs: {len(ri_pairs)}")
print(f"  n_paths_c0 = 0 (SIM-bridged only): {(ri_pairs['n_paths_c0'] == 0).sum()}")

# ─── Build name lookup ────────────────────────────────��───────────────────────
risk_name_map = dict(
    zip(risk_names["cluster_id"].astype(str), risk_names["cluster_name"].str[:80])
)
interv_name_map = dict(
    zip(interv_names["cluster_id"].astype(str), interv_names["cluster_name"].str[:80])
)
interv_size_map = dict(
    zip(interv_names["cluster_id"].astype(str), interv_names["n_nodes"])
)

# ─── Classify intervention clusters by type ───────────────────────��───────────
# Field-building interventions (most generic — funded by any risk, least novel):
# - Cluster 8: Fund and expand AI safety research teams (I8 — dominant meta-intervention)
# - Cluster 11: Produce and share accessible AI x-risk educational content
# - Cluster 16: (check) — need to look at all names
FIELD_BUILDING_CLUSTERS = set()
GOVERNANCE_CLUSTERS = set()
TECHNICAL_CLUSTERS = set()

FIELD_BUILDING_KEYWORDS = [
    "fund",
    "educat",
    "outreach",
    "train researchers",
    "research community",
    "fellowship",
    "grant",
    "scholarship",
    "workshop",
    "careers in ai safety",
    "career",
    "recruit",
    "mentorship",
    "attract talent",
]
GOVERNANCE_KEYWORDS = [
    "regulat",
    "govern",
    "policy",
    "treaty",
    "standard",
    "framework",
    "export control",
    "international",
    "legislation",
    "agreement",
    "oversight",
    "audit",
    "certif",
    "license",
]

for _, row in interv_names.iterrows():
    cid = str(row["cluster_id"])
    name_lower = str(row["cluster_name"]).lower()
    if any(kw in name_lower for kw in FIELD_BUILDING_KEYWORDS):
        FIELD_BUILDING_CLUSTERS.add(cid)
    elif any(kw in name_lower for kw in GOVERNANCE_KEYWORDS):
        GOVERNANCE_CLUSTERS.add(cid)
    else:
        TECHNICAL_CLUSTERS.add(cid)

print("\nIntervention cluster types (based on name keywords):")
print(
    f"  Field-building: {sorted(FIELD_BUILDING_CLUSTERS)} ({len(FIELD_BUILDING_CLUSTERS)} clusters)"
)
print(
    f"  Governance: {sorted(GOVERNANCE_CLUSTERS)} ({len(GOVERNANCE_CLUSTERS)} clusters)"
)
print(f"  Technical: {sorted(TECHNICAL_CLUSTERS)} ({len(TECHNICAL_CLUSTERS)} clusters)")

# ─── Filter to SIM-bridged-only pairs ───────────────────────────────���────────
sim_bridged = ri_pairs[ri_pairs["n_paths_c0"] == 0].copy()
sim_bridged["risk_cid_str"] = sim_bridged["risk_cid"].astype(int).astype(str)
sim_bridged["interv_cid_str"] = sim_bridged["interv_cid"].astype(int).astype(str)

sim_bridged["risk_name"] = (
    sim_bridged["risk_cid_str"].map(risk_name_map).fillna("unknown")
)
sim_bridged["interv_name"] = (
    sim_bridged["interv_cid_str"].map(interv_name_map).fillna("unknown")
)
sim_bridged["interv_type"] = sim_bridged["interv_cid_str"].apply(
    lambda c: "field-building"
    if c in FIELD_BUILDING_CLUSTERS
    else ("governance" if c in GOVERNANCE_CLUSTERS else "technical")
)
sim_bridged["interv_n_nodes"] = (
    sim_bridged["interv_cid_str"].map(interv_size_map).fillna(0)
)

print(f"\nSIM-bridged-only pairs ({len(sim_bridged)} total):")
print(sim_bridged["interv_type"].value_counts())

# ─── Full list sorted by n_paths_c1 ─────────────────────────���────────────────
sim_bridged_sorted = sim_bridged.sort_values("n_paths_c1", ascending=False).reset_index(
    drop=True
)

# Save full list
full_path = os.path.join(CONN_DIR, "novel_sim_bridged_pairs_full.csv")
sim_bridged_sorted[
    [
        "risk_cid",
        "interv_cid",
        "risk_name",
        "interv_name",
        "interv_type",
        "n_paths_c0",
        "n_paths_c1",
        "n_paths_c2",
        "c1_boost",
    ]
].to_csv(full_path, index=False)
print(f"\nSaved full list: {full_path} ({len(sim_bridged_sorted)} rows)")

# ─── Apply novelty filter: exclude field-building interventions ──────────────
novel_pairs = sim_bridged_sorted[
    sim_bridged_sorted["interv_type"] != "field-building"
].reset_index(drop=True)
print(f"\nAfter novelty filter (exclude field-building): {len(novel_pairs)} pairs")

# Top-20
top20 = novel_pairs.head(20).copy()
top20["rank"] = range(1, len(top20) + 1)
top20["novelty_note"] = ""  # Placeholder for manual web-search annotation
top20["post_2023_paper"] = "TBD"  # To be filled in manually

print("\n=== Top-20 Novel SIM-bridged-only R→I Pairs (post novelty filter) ===")
print(f"{'Rank':4} {'Type':12} {'c1 paths':8} {'Risk cluster':40} {'Intervention':40}")
print("-" * 110)
for _, row in top20.iterrows():
    print(
        f"{row['rank']:4} {row['interv_type']:12} {int(row['n_paths_c1']):8} "
        f"{row['risk_name'][:38]:40} {row['interv_name'][:38]:40}"
    )

# Save top-20
top20_path = os.path.join(CONN_DIR, "novel_sim_bridged_pairs_top20.csv")
top20[
    [
        "rank",
        "risk_cid",
        "interv_cid",
        "interv_type",
        "risk_name",
        "interv_name",
        "n_paths_c1",
        "n_paths_c2",
        "c1_boost",
        "novelty_note",
        "post_2023_paper",
    ]
].to_csv(top20_path, index=False)
print(f"\nSaved top-20: {top20_path}")

# ─── Summary statistics ────────────────────────���─────────────────────────���────
print("\n=== Summary: SIM-bridged-only connections ===")
total_pairs = len(ri_pairs)
bridged_only = len(sim_bridged)
after_filter = len(novel_pairs)
print(f"  Total R→I cluster pairs: {total_pairs}")
print(
    f"  SIM-bridged-only (n_paths_c0=0): {bridged_only} ({100 * bridged_only / total_pairs:.1f}%)"
)
print(f"  After novelty filter (non-field-building): {after_filter}")
print(f"  Of which technical: {(novel_pairs['interv_type'] == 'technical').sum()}")
print(f"  Of which governance: {(novel_pairs['interv_type'] == 'governance').sum()}")
print("\n  Top novel pair by c1 paths:")
if len(novel_pairs) > 0:
    top1 = novel_pairs.iloc[0]
    print(f"    Risk: {top1['risk_name']}")
    print(f"    Intervention: {top1['interv_name']} [{top1['interv_type']}]")
    print(
        f"    n_paths_c1: {int(top1['n_paths_c1'])}, n_paths_c2: {int(top1['n_paths_c2'])}"
    )

print("\nDone.")
