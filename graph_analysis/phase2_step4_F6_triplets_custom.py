"""
phase2_step4_F6_triplets_custom.py [rev8]

E5-equivalent for rev8: rebuilds R->Group->I triplets using the new
custom-consim1 frozenset group taxonomy.

Inputs:
  step4_connectivity/ri_triplets_custom_consim1.csv  -- base triplets (bfamily_id level)
  step4_cluster_tables/frozenset_group_memberships_custom_consim1.csv
  step5_naming/frozenset_group_names_custom_llm.csv
  step5_naming/risk_cluster_names_llm_v2.csv
  step5_naming/intervention_cluster_names_llm_v2.csv

Outputs:
  step4_connectivity/ri_triplets_custom_consim1_rev8.csv -- base triplets with group_id added
  step4_connectivity/ri_group_triplets_custom_consim1.csv -- aggregated R->Group->I triplets
  step4_connectivity/ri_group_triplets_top20_custom_consim1.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "phase2_results"
STEP4_DIR = RESULTS_DIR / "step4_finalanalysis"
CONN_DIR = STEP4_DIR / "step4_connectivity"
TABLES_DIR = STEP4_DIR / "step4_cluster_tables"
NAMING_DIR = RESULTS_DIR / "step5_naming"

tri = pd.read_csv(CONN_DIR / "ri_triplets_custom_consim1.csv")
memberships = pd.read_csv(TABLES_DIR / "frozenset_group_memberships_custom_consim1.csv")
group_names = pd.read_csv(NAMING_DIR / "frozenset_group_names_custom_llm.csv")
risk_df = pd.read_csv(NAMING_DIR / "risk_cluster_names_llm_v2.csv")
interv_df = pd.read_csv(NAMING_DIR / "intervention_cluster_names_llm_v2.csv")

print(f"Base triplets: {len(tri)} rows")
print(
    f"Memberships: {len(memberships)} rows, {memberships['group_id'].nunique()} groups"
)

fam_to_group = dict(
    zip(memberships["family_id"].astype(int), memberships["group_id"].astype(int))
)
group_name_map = dict(
    zip(group_names["group_id"].astype(int), group_names["final_name"].astype(str))
)

risk_col = "final_name" if "final_name" in risk_df.columns else "llm_name"
interv_col = "final_name" if "final_name" in interv_df.columns else "llm_name"
risk_name_map = dict(
    zip(risk_df["cluster_id"].astype(int), risk_df[risk_col].astype(str))
)
interv_name_map = dict(
    zip(interv_df["cluster_id"].astype(int), interv_df[interv_col].astype(str))
)

tri["group_id"] = tri["bfamily_id"].astype(int).map(fam_to_group)
n_missing = tri["group_id"].isna().sum()
print(f"Frozensets without group assignment: {n_missing}")

tri["group_name"] = tri["group_id"].map(group_name_map)
tri_out = tri.copy()
tri_out.to_csv(CONN_DIR / "ri_triplets_custom_consim1_rev8.csv", index=False)
print(f"Written: ri_triplets_custom_consim1_rev8.csv ({len(tri_out)} rows)")

agg = (
    tri.dropna(subset=["group_id"])
    .groupby(["risk_cid", "group_id", "interv_cid"])["n_triplet_paths"]
    .sum()
    .reset_index()
    .sort_values("n_triplet_paths", ascending=False)
    .reset_index(drop=True)
)

agg["risk_name"] = agg["risk_cid"].astype(int).map(risk_name_map)
agg["group_name"] = agg["group_id"].astype(int).map(group_name_map)
agg["interv_name"] = agg["interv_cid"].astype(int).map(interv_name_map)

agg = agg[
    [
        "risk_cid",
        "risk_name",
        "group_id",
        "group_name",
        "interv_cid",
        "interv_name",
        "n_triplet_paths",
    ]
]

agg.to_csv(CONN_DIR / "ri_group_triplets_custom_consim1.csv", index=False)
print(f"Written: ri_group_triplets_custom_consim1.csv ({len(agg)} rows)")

top20 = agg.head(20)
top20.to_csv(CONN_DIR / "ri_group_triplets_top20_custom_consim1.csv", index=False)
print("Written: ri_group_triplets_top20_custom_consim1.csv (20 rows)")

total_orig = tri["n_triplet_paths"].sum()
total_agg = agg["n_triplet_paths"].sum()
print("\nQuality:")
print(f"  Total paths original: {total_orig}")
print(f"  Total paths aggregated: {total_agg}")
print(f"  Match: {total_orig == total_agg}")
print(f"  Unique (risk, group, interv) triplets: {len(agg)}")
print(f"  Unique groups appearing: {agg['group_id'].nunique()}")

print("\nTop-20 R->Group->I triplets (rev8):")
for i, r in top20.iterrows():
    rn = str(r["risk_name"])[:35] if pd.notna(r["risk_name"]) else "?"
    gn = str(r["group_name"])[:40] if pd.notna(r["group_name"]) else "?"
    inv = str(r["interv_name"])[:35] if pd.notna(r["interv_name"]) else "?"
    print(
        f"  {i + 1:2d}. [{rn}] --[{gn}]--> [{inv}]  ({int(r['n_triplet_paths'])} paths)"
    )
