"""
Item 11 follow-on: Meta-cluster level R→C→I triplet rollup.

Rolls up ri_triplets_consim1.csv (L1/L3 cluster-level) to
(risk_meta_k10, pathbuildB_metafamily, interv_meta_k10) triplets.

Outputs:
  step4_connectivity/ri_meta_triplets_consim1.csv
  step4_connectivity/ri_meta_triplets_histogram.png
  step4_connectivity/ri_meta_triplets_top20.csv  — top-20 by n_triplet_paths
"""

from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
CONN = BASE / "phase2_results/step4_finalanalysis/step4_connectivity"
META = BASE / "phase2_results/step4_finalanalysis/step4_metaclusters"
TABLES = BASE / "phase2_results/step4_finalanalysis/step4_cluster_tables"
NAMING = BASE / "phase2_results/step5_naming"

# ── Load source tables ─────────────────────────────────────────────────────
triplets = pd.read_csv(CONN / "ri_triplets_consim1.csv")
print(f"L1/L3 triplets loaded: {len(triplets):,}")

risk_meta = pd.read_csv(META / "risk_meta_assignments.csv")
interv_meta = pd.read_csv(META / "intervention_meta_assignments.csv")
bfam_meta = pd.read_csv(TABLES / "pathbuildB_metafamilies_consim1.csv")[
    ["family_id", "meta_family_id"]
]
mf_summary = pd.read_csv(TABLES / "pathbuildB_metafamily_summary_consim1.csv")

# v2 names for B-family display
try:
    v2_df = pd.read_csv(NAMING / "pathbuildB_chain_names_llm_v2.csv")
    fam_v2 = dict(zip(v2_df["cluster_id"], v2_df["final_name"]))
except Exception:
    fam_v2 = {}

# ── Build meta-cluster name lookup (dominant cluster by n_nodes within meta) ──
# Risk meta names
risk_meta_names = (
    risk_meta.sort_values("n_nodes", ascending=False)
    .drop_duplicates("meta_k10")[["meta_k10", "cluster_name"]]
    .rename(columns={"meta_k10": "risk_meta_id", "cluster_name": "risk_meta_name"})
)

# Intervention meta names (use meta_k10 column)
interv_meta_names = (
    interv_meta.sort_values("n_nodes", ascending=False)
    .drop_duplicates("meta_k10")[["meta_k10", "cluster_name"]]
    .rename(columns={"meta_k10": "interv_meta_id", "cluster_name": "interv_meta_name"})
)

# Meta-family names (dominant_family_name from summary)
mf_names = dict(zip(mf_summary["meta_family_id"], mf_summary["dominant_family_name"]))
mf_paths = dict(zip(mf_summary["meta_family_id"], mf_summary["n_paths_total"]))

# ── Join triplets with meta assignments ────────────────────────────────────
df = triplets.copy()
df = df.merge(
    risk_meta[["cluster_id", "meta_k10"]].rename(
        columns={"cluster_id": "risk_cid", "meta_k10": "risk_meta_id"}
    ),
    on="risk_cid",
    how="left",
)
df = df.merge(
    interv_meta[["cluster_id", "meta_k10"]].rename(
        columns={"cluster_id": "interv_cid", "meta_k10": "interv_meta_id"}
    ),
    on="interv_cid",
    how="left",
)
df = df.merge(
    bfam_meta.rename(columns={"family_id": "bfamily_id"}), on="bfamily_id", how="left"
)

print(
    f"After join — rows with risk_meta: {df['risk_meta_id'].notna().sum():,}, "
    f"interv_meta: {df['interv_meta_id'].notna().sum():,}, "
    f"meta_family: {df['meta_family_id'].notna().sum():,}"
)

# ── Roll up to (risk_meta, meta_family, interv_meta) ──────────────────────
rollup = (
    df.dropna(subset=["risk_meta_id", "meta_family_id", "interv_meta_id"])
    .groupby(["risk_meta_id", "meta_family_id", "interv_meta_id"], as_index=False)[
        "n_triplet_paths"
    ]
    .sum()
)
rollup["risk_meta_id"] = rollup["risk_meta_id"].astype(int)
rollup["meta_family_id"] = rollup["meta_family_id"].astype(int)
rollup["interv_meta_id"] = rollup["interv_meta_id"].astype(int)
rollup = rollup.sort_values("n_triplet_paths", ascending=False).reset_index(drop=True)

# Add names
rollup = rollup.merge(risk_meta_names, on="risk_meta_id", how="left")
rollup = rollup.merge(interv_meta_names, on="interv_meta_id", how="left")
rollup["meta_family_name"] = rollup["meta_family_id"].map(mf_names)

# Column order
rollup = rollup[
    [
        "risk_meta_id",
        "risk_meta_name",
        "meta_family_id",
        "meta_family_name",
        "interv_meta_id",
        "interv_meta_name",
        "n_triplet_paths",
    ]
]

print(f"\n{len(rollup):,} meta-triplets total")
print("Top-10 meta-triplets by path count:")
for _, r in rollup.head(10).iterrows():
    print(
        f"  R-meta{int(r['risk_meta_id'])} × MF{int(r['meta_family_id'])} × I-meta{int(r['interv_meta_id'])}: "
        f"{int(r['n_triplet_paths']):,} paths"
    )
    print(f"    Risk: {str(r['risk_meta_name'])[:60]}")
    print(f"    Chain: {str(r['meta_family_name'])[:60]}")
    print(f"    Interv: {str(r['interv_meta_name'])[:60]}")

# Save
out_csv = CONN / "ri_meta_triplets_consim1.csv"
rollup.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv} ({len(rollup):,} rows)")

top20 = rollup.head(20)
out_top20 = CONN / "ri_meta_triplets_top20_consim1.csv"
top20.to_csv(out_top20, index=False)
print(f"Saved {out_top20}")

# ── Histogram ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of all meta-triplet path counts (log y, linear x)
ax = axes[0]
ax.hist(rollup["n_triplet_paths"], bins=50, color="steelblue", edgecolor="white")
ax.set_yscale("log")
ax.set_xlabel("n_triplet_paths per meta-triplet")
ax.set_ylabel("Count (log scale)")
ax.set_title(
    f"Meta-cluster triplet path count distribution\n({len(rollup):,} triplets, consim1)"
)

# Right: top-20 meta-triplets horizontal bar
ax = axes[1]
top20_plot = rollup.head(20).copy()
top20_plot["label"] = (
    "R"
    + top20_plot["risk_meta_id"].astype(str)
    + " × MF"
    + top20_plot["meta_family_id"].astype(str)
    + " × I"
    + top20_plot["interv_meta_id"].astype(str)
)
top20_plot = top20_plot.sort_values("n_triplet_paths")
ax.barh(
    range(len(top20_plot)), top20_plot["n_triplet_paths"], color="steelblue", height=0.7
)
ax.set_yticks(range(len(top20_plot)))
ax.set_yticklabels(top20_plot["label"], fontsize=8)
ax.set_xlabel("n_triplet_paths")
ax.set_title("Top-20 meta-cluster R→B-metafamily→I triplets (consim1)")

plt.tight_layout()
out_hist = CONN / "ri_meta_triplets_histogram.png"
fig.savefig(out_hist, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_hist}")
print("\nDone — Item 11 meta-triplets complete.")
