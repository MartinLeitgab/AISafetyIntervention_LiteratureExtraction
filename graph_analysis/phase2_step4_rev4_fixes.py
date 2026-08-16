"""
Rev4 fixes — Items 4, 6, 7, 10 from step45_revision3_inputs.txt

Item 4:  PathbuildB meta-family 2D MDS plot (like risk/interv 2D MDS)
Item 6:  Fix pathbuildB_family_size_distribution.png — log-y, standard log x-axis (1,10,100)
Item 7:  Update pathbuildB_metafamily_dendrogram.png with proper GPT names (no B167 labels)
Item 10: Fix CSV truncation in optionB_top20_decoded_consim*.csv — full text in all columns
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

BASE = Path(__file__).parent
STEP4 = BASE / "phase2_results/step4_finalanalysis"
META_DIR = STEP4 / "step4_metaclusters"
CONN_DIR = STEP4 / "step4_connectivity"
TABLE_DIR = STEP4 / "step4_cluster_tables"
NAMING_DIR = BASE / "phase2_results/step5_naming"

print("=== Rev4 fixes ===\n")


# ══════════════════════════════════════════════════════════════════════════════
# Item 6 — Fix family size distribution: log-y, standard log x-axis
# ══════════════════════════════════════════════════════════════════════════════
print("Item 6 — PathbuildB family size distribution (log-y, standard log x-axis) …")

fam_df = pd.read_csv(TABLE_DIR / "optionB_cooccurrence_families_consim1.csv")
n_paths_all = fam_df["n_paths"].values
n_paths_ge5 = n_paths_all[n_paths_all >= 5]

n_ge5 = (n_paths_all >= 5).sum()
n_ge10 = (n_paths_all >= 10).sum()
n_ge100 = (n_paths_all >= 100).sum()
print(f"  n_families={len(fam_df)}, n≥5={n_ge5}, n≥10={n_ge10}, n≥100={n_ge100}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: linear-scale histogram for n≥5 families
axes[0].hist(n_paths_ge5, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("N paths per family")
axes[0].set_ylabel("Number of families")
axes[0].set_title(f"Family Size — linear scale (n≥5 families, n={n_ge5})")
med = float(np.median(n_paths_ge5))
axes[0].axvline(med, color="red", lw=1.5, linestyle="--", label=f"median={med:.0f}")
axes[0].legend(fontsize=9)
axes[0].set_yscale("log")

# Right: proper log-x axis (standard ticks 1, 10, 100, 1000, 10000)
# Use standard log-scale x-axis with log-y
axes[1].hist(
    n_paths_all,
    bins=np.logspace(0, 4, 60),
    color="darkorange",
    edgecolor="white",
    alpha=0.85,
)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("N paths per family (log scale)")
axes[1].set_ylabel("Number of families (log scale)")
axes[1].set_title(f"Family Size — log-log scale (all {len(fam_df)} families)")
# Standard tick labels (no exponent notation)
axes[1].xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"
    )
)
for threshold, label in [(5, "n=5"), (10, "n=10"), (100, "n=100"), (1000, "n=1000")]:
    axes[1].axvline(threshold, color="gray", lw=1.2, linestyle=":", alpha=0.7)
    axes[1].text(
        threshold * 1.05,
        axes[1].get_ylim()[1] * 0.7 if axes[1].get_ylim()[1] > 1 else 2,
        label,
        fontsize=7,
        color="gray",
    )

plt.suptitle(
    f"PathbuildB Co-occurrence Family Size Distribution (consim1, 1,603 total families)\n"
    f"n≥5: {n_ge5} ({100 * n_ge5 / len(fam_df):.0f}%)   "
    f"n≥10: {n_ge10} ({100 * n_ge10 / len(fam_df):.0f}%)   "
    f"n≥100: {n_ge100} ({100 * n_ge100 / len(fam_df):.0f}%)",
    fontsize=11,
)
plt.tight_layout()
out = TABLE_DIR / "pathbuildB_family_size_distribution.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Item 7 — Update pathbuildB_metafamily_dendrogram.png with proper names
# ══════════════════════════════════════════════════════════════════════════════
print("\nItem 7 — Regenerate pathbuildB metafamily dendrogram with GPT names …")

summary = pd.read_csv(TABLE_DIR / "pathbuildB_metafamily_summary_consim1.csv")
families = pd.read_csv(TABLE_DIR / "pathbuildB_metafamilies_consim1.csv")

# Build full meta-family name map
mf_name = dict(zip(summary["meta_family_id"], summary["dominant_family_name"]))
mf_paths = dict(zip(summary["meta_family_id"], summary["n_paths_total"]))
mf_n_fam = dict(zip(summary["meta_family_id"], summary["n_families"]))

mf_ids = sorted(summary["meta_family_id"].unique())

# Compute pairwise Jaccard distance between meta-families
# Represent each meta-family as a frozenset of member family_ids
mf_members = {}
for mf_id in mf_ids:
    member_fam_ids = set(
        families[families["meta_family_id"] == mf_id]["family_id"].tolist()
    )
    mf_members[mf_id] = member_fam_ids

# Also use component-level frozensets (the actual Jaccard used in D1)
# Parse family signatures into frozensets
fam_sig = {}
for _, row in families.iterrows():
    sig = frozenset(
        c.strip() for c in str(row["signature_str"]).split(" & ") if ":" in c
    )
    fam_sig[row["family_id"]] = sig

mf_sigs = {}
for mf_id in mf_ids:
    all_fam = families[families["meta_family_id"] == mf_id]["family_id"].tolist()
    # Union of all component frozensets in this meta-family
    union = frozenset()
    for fid in all_fam:
        union = union | fam_sig.get(fid, frozenset())
    mf_sigs[mf_id] = union

# Pairwise Jaccard similarity
n = len(mf_ids)
jac_sim = np.zeros((n, n))
for i, a in enumerate(mf_ids):
    for j, b in enumerate(mf_ids):
        sa, sb = mf_sigs[a], mf_sigs[b]
        inter = len(sa & sb)
        union_len = len(sa | sb)
        jac_sim[i, j] = inter / union_len if union_len > 0 else 0.0

dist_mat = 1.0 - jac_sim
np.fill_diagonal(dist_mat, 0.0)
dist_mat = np.clip(dist_mat, 0, None)
condensed = squareform(dist_mat, checks=False)
Z = linkage(condensed, method="average")

fig, ax = plt.subplots(figsize=(18, 10))
dend = dendrogram(
    Z,
    labels=[
        f"MF{mid}\n{mf_name.get(mid, '?')}\n({mf_paths.get(mid, 0):,} paths, {mf_n_fam.get(mid, 0)} fam)"
        for mid in mf_ids
    ],
    leaf_rotation=0,
    orientation="top",
    ax=ax,
    color_threshold=0.0,
    above_threshold_color="steelblue",
)

ax.set_ylabel("1 − Jaccard similarity (component union)", fontsize=11)
ax.set_title(
    "PathbuildB Meta-Family Dendrogram (k=32, average linkage on 1-Jaccard component similarity)\n"
    "Each meta-family represented by union of its constituent family component frozensets",
    fontsize=12,
)
plt.xticks(fontsize=6.5, rotation=90)
plt.tight_layout()
out_dend = TABLE_DIR / "pathbuildB_metafamily_dendrogram.png"
fig.savefig(out_dend, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_dend.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Item 4 — PathbuildB meta-family 2D MDS plot
# ══════════════════════════════════════════════════════════════════════════════
print("\nItem 4 — PathbuildB meta-family 2D MDS …")

# Project all 32 meta-families into 2D using Jaccard distance
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, max_iter=3000)
coords = mds.fit_transform(dist_mat)

# Color by path count (log scale)
path_counts = np.array([mf_paths.get(mid, 1) for mid in mf_ids])
log_paths = np.log10(path_counts.clip(1))

fig, ax = plt.subplots(figsize=(16, 12))
sc = ax.scatter(
    coords[:, 0],
    coords[:, 1],
    c=log_paths,
    cmap="plasma",
    s=[80 + 200 * (lp / log_paths.max()) for lp in log_paths],
    edgecolors="black",
    linewidths=0.5,
    zorder=3,
    alpha=0.85,
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
cbar.set_label("log10(n_paths)", fontsize=10)

for i, mid in enumerate(mf_ids):
    name = mf_name.get(mid, f"MF{mid}")
    # Wrap long names
    wrapped = "\n".join(textwrap.wrap(name, 28))
    ax.annotate(
        f"MF{mid}\n{wrapped}",
        (coords[i, 0], coords[i, 1]),
        fontsize=6.5,
        ha="center",
        va="bottom",
        xytext=(0, 8),
        textcoords="offset points",
        color="black",
    )

ax.set_xlabel("MDS dim 1 (Jaccard distance)", fontsize=11)
ax.set_ylabel("MDS dim 2", fontsize=11)
ax.set_title(
    "PathbuildB Meta-Families — 2D MDS by Jaccard Component Distance (k=32)\n"
    "Point size ∝ log(n_paths); color = log10(n_paths); layout preserves pairwise Jaccard distances",
    fontsize=12,
)
plt.tight_layout()
out_mds = TABLE_DIR / "pathbuildB_metafamily_2d_mds.png"
fig.savefig(out_mds, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_mds.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Item 10 — Fix truncated text in optionB_top20_decoded_consim*.csv
# ══════════════════════════════════════════════════════════════════════════════
print("\nItem 10 — Fix CSV truncation in optionB_top20_decoded_consim*.csv …")

# Load cluster/subtype representatives for full labels
reps = pd.read_csv(TABLE_DIR / "bodysubtype_cluster_representatives.csv")
print(f"  Representatives columns: {reps.columns.tolist()}")
print(f"  Representatives shape: {reps.shape}")
print(f"  Sample: {reps.head(2).to_string()}")

# Build full name lookup: prefix_key → full rep_name
prefix_to_full_name = dict(zip(reps["prefix_key"], reps["rep_name"]))


def decode_full(sig_str: str) -> str:
    """Decode signature string to full (non-truncated) component descriptions."""
    parts = [p.strip() for p in sig_str.split("&")]
    lines = []
    for part in parts:
        full_name = prefix_to_full_name.get(part, f"[{part}]")
        lines.append(f"{part}: {full_name}")
    return "\n".join(lines)


# Semantic ordering: pr → th → de → im → va
SEMANTIC_ORDER = {"pr": 0, "th": 1, "de": 2, "im": 3, "va": 4}


def reorder_and_decode_full(sig_str: str) -> str:
    """Decode + reorder in semantic order (pr→th→de→im→va), full text."""
    parts = sorted(
        [p.strip() for p in sig_str.split("&")],
        key=lambda p: (SEMANTIC_ORDER.get(p.split(":")[0], 9), p),
    )
    lines = []
    for part in parts:
        full_name = prefix_to_full_name.get(part, f"[{part}]")
        lines.append(f"{part}: {full_name}")
    return "\n".join(lines)


for consim in [0, 1, 2]:
    fname = f"optionB_top20_decoded_consim{consim}.csv"
    fpath = TABLE_DIR / fname
    if not fpath.exists():
        print(f"  SKIP {fname} (not found)")
        continue
    df = pd.read_csv(fpath)
    # Rebuild decoded_chain_components from signature_str with full names
    df["decoded_chain_components"] = df["signature_str"].apply(reorder_and_decode_full)
    df.to_csv(fpath, index=False)
    print(f"  Fixed {fname}: {len(df)} rows, no truncation")

print("  Item 10 complete.")
print("\n=== Rev4 fixes complete ===")
