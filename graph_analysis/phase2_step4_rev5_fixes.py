"""
Rev5 fixes — Items 3, 4, 8, 10 from step45_revision4_inputs.txt

Item 3:  Fix truncation in novel_sim_bridged_pairs_top20_technical.csv
         (B5 now reads v2 naming CSVs; risk_name/interv_name use full final_name)
Item 4:  Replace non-rendering special characters (→, –, curly quotes) in all
         step4 and step5 output CSVs
Item 8:  Fix pathbuildB_metafamily_2d_mds.png colorbar — show actual path counts
         (1, 10, 100, 1000, 10000) instead of log10 exponent values
Item 10: Fix pathbuildB_metafamily_dendrogram.png — orientation="left" so MF titles
         are horizontal and 1-Jaccard similarity is the x-axis
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
from matplotlib.colors import LogNorm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

BASE = Path(__file__).parent
STEP4 = BASE / "phase2_results/step4_finalanalysis"
META_DIR = STEP4 / "step4_metaclusters"
CONN_DIR = STEP4 / "step4_connectivity"
TABLE_DIR = STEP4 / "step4_cluster_tables"
NAMING_DIR = BASE / "phase2_results/step5_naming"
STEP5_DIR = BASE / "phase2_results/step5_naming"

print("=== Rev5 fixes ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# Item 3 — Fix name truncation in novel_sim_bridged_pairs CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("Item 3 — Fix truncation: novel_sim_bridged_pairs CSVs …")

# Load v2 naming with full final_names (no truncation)
risk_names_df = pd.read_csv(NAMING_DIR / "risk_cluster_names_llm_v2.csv")
interv_names_df = pd.read_csv(NAMING_DIR / "intervention_cluster_names_llm_v2.csv")

risk_name_col = "final_name" if "final_name" in risk_names_df.columns else "llm_name"
interv_name_col = (
    "final_name" if "final_name" in interv_names_df.columns else "llm_name"
)

risk_name_map = dict(
    zip(risk_names_df["cluster_id"].astype(str), risk_names_df[risk_name_col])
)
interv_name_map = dict(
    zip(interv_names_df["cluster_id"].astype(str), interv_names_df[interv_name_col])
)

# Check top10 names for visibility
print(f"  Risk name map sample: R10={risk_name_map.get('10', '?')}")
print(f"  Interv name map sample: I26={interv_name_map.get('26', '?')}")

# Regenerate novel_sim_bridged_pairs CSVs from cross_config_ri_pairs.csv
ri_pairs_path = CONN_DIR / "cross_config_ri_pairs.csv"
if ri_pairs_path.exists():
    ri_pairs = pd.read_csv(ri_pairs_path)
    ri_pairs["risk_cid_str"] = ri_pairs["risk_cid"].astype(int).astype(str)
    ri_pairs["interv_cid_str"] = ri_pairs["interv_cid"].astype(int).astype(str)
    ri_pairs["risk_name"] = (
        ri_pairs["risk_cid_str"].map(risk_name_map).fillna("unknown")
    )
    ri_pairs["interv_name"] = (
        ri_pairs["interv_cid_str"].map(interv_name_map).fillna("unknown")
    )

    # ── classify intervention clusters ──────────────────────────────────────
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

    def classify(name_lower):
        if any(kw in name_lower for kw in FIELD_BUILDING_KEYWORDS):
            return "field-building"
        if any(kw in name_lower for kw in GOVERNANCE_KEYWORDS):
            return "governance"
        return "technical"

    ri_pairs["interv_type"] = ri_pairs["interv_cid_str"].apply(
        lambda c: classify(str(interv_name_map.get(c, "")).lower())
    )
    interv_size_map = dict(
        zip(interv_names_df["cluster_id"].astype(str), interv_names_df["n_members"])
    )
    ri_pairs["interv_n_nodes"] = (
        ri_pairs["interv_cid_str"].map(interv_size_map).fillna(0)
    )

    # SIM-bridged-only pairs
    sim_bridged = ri_pairs[ri_pairs["n_paths_c0"] == 0].copy()
    sim_bridged_sorted = sim_bridged.sort_values(
        "n_paths_c1", ascending=False
    ).reset_index(drop=True)

    # Full list
    full_path = CONN_DIR / "novel_sim_bridged_pairs_full.csv"
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
    print(f"  Saved full list: {full_path.name} ({len(sim_bridged_sorted)} rows)")

    # Filter and top-20
    novel_pairs = sim_bridged_sorted[
        sim_bridged_sorted["interv_type"] != "field-building"
    ].reset_index(drop=True)
    top20 = novel_pairs.head(20).copy()
    top20["rank"] = range(1, len(top20) + 1)

    top20_path = CONN_DIR / "novel_sim_bridged_pairs_top20.csv"
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
        ]
    ].to_csv(top20_path, index=False)
    print(f"  Saved top-20: {top20_path.name} ({len(top20)} rows)")

    # ── Update technical CSV: replace risk_name and interv_name columns only ──
    tech_path = CONN_DIR / "novel_sim_bridged_pairs_top20_technical.csv"
    if tech_path.exists():
        tech_df = pd.read_csv(tech_path)
        # Rebuild names from v2 mapping using the cluster IDs already in the file
        tech_df["risk_name"] = tech_df["risk_cid"].apply(
            lambda c: risk_name_map.get(str(int(c)), "unknown")
        )
        tech_df["interv_name"] = tech_df["interv_cid"].apply(
            lambda c: interv_name_map.get(str(int(c)), "unknown")
        )
        tech_df.to_csv(tech_path, index=False)
        print(
            f"  Updated {tech_path.name}: replaced risk_name/interv_name with full v2 names"
        )
        # Show sample
        for _, r in tech_df.head(3).iterrows():
            print(
                f"    R{int(r['risk_cid'])}→I{int(r['interv_cid'])}: {r['risk_name']} | {r['interv_name']}"
            )
    else:
        print(f"  NOTE: {tech_path.name} not found — only regenerated top20.csv")
else:
    print(f"  WARNING: {ri_pairs_path} not found — skipping Item 3")

print("  Item 3 complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Item 4 — Replace non-rendering special characters in all step4/5 CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("Item 4 — Replacing special characters in all step4/step5 CSVs …")

REPLACEMENTS = [
    ("\u2192", "->"),  # → (right arrow)
    ("\u2013", "-"),  # – (en dash)
    ("\u2014", "--"),  # — (em dash)
    ("\u2018", "'"),  # ' (left single quote)
    ("\u2019", "'"),  # ' (right single quote)
    ("\u201c", '"'),  # " (left double quote)
    ("\u201d", '"'),  # " (right double quote)
    ("\u2026", "..."),  # … (ellipsis)
    ("\u00e2\u0080\u0099", "'"),  # UTF-8 re-encoded ' artifacts
    ("\u00e2\u0080\u009c", '"'),  # UTF-8 re-encoded " artifacts
    ("\u00e2\u0080\u009d", '"'),  # UTF-8 re-encoded " artifacts
    ("\u00e2\u0086\u0092", "->"),  # UTF-8 re-encoded → artifacts
    ("\u00e2\u0080\u0093", "-"),  # UTF-8 re-encoded – artifacts
    ("\u00e2\u0080\u0094", "--"),  # UTF-8 re-encoded — artifacts
    ("\u00e2\u0080\u00a6", "..."),  # UTF-8 re-encoded … artifacts
    ("\u2248", "~"),  # ≈
    ("\u2265", ">="),  # ≥
    ("\u2264", "<="),  # ≤
    ("\u00d7", "x"),  # ×
    ("\u00e9", "e"),  # é
    ("\u00e8", "e"),  # è
    ("\u00fc", "ue"),  # ü
    ("\u03b1", "alpha"),  # α
    ("\u03b2", "beta"),  # β
    ("\u03bb", "lambda"),  # λ
    ("\u2113", "l"),  # ℓ
]


def clean_text(text):
    if not isinstance(text, str):
        return text
    for bad, good in REPLACEMENTS:
        text = text.replace(bad, good)
    # Also handle wrongly decoded multi-byte sequences (Latin-1 misread of UTF-8)
    text = text.replace("\u00e2\u0086\u0092", "->")  # â†' = →
    text = text.replace("\u00e2\u0080\u0099", "'")  # â€™ = '
    text = text.replace("\u00e2\u0080\u009c", '"')  # â€œ = "
    text = text.replace("\u00e2\u0080\u009d", '"')  # â€ = "
    text = text.replace("\u00e2\u0080\u0094", "--")  # â€" = —
    text = text.replace("\u00e2\u0080\u0093", "-")  # â€" = –
    text = text.replace("\u00e2\u0080\u00a6", "...")  # â€¦ = …
    return text


# Collect all CSV files in step4 and step5 output dirs
csv_dirs = [
    STEP4,
    STEP4 / "step4_cluster_tables",
    STEP4 / "step4_connectivity",
    STEP4 / "step4_metaclusters",
    STEP4 / "step4_paths",
    STEP4 / "step4_subclusters",
    BASE / "phase2_results/step5_naming",
]

total_fixed = 0
total_unchanged = 0

for d in csv_dirs:
    if not d.exists():
        continue
    for csv_path in sorted(d.glob("*.csv")):
        # Skip archive folders
        if "archive" in str(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, dtype=str)
            original_str = df.to_csv(index=False)
            # Apply cleaning to all string columns
            df = df.applymap(clean_text)
            cleaned_str = df.to_csv(index=False)
            if cleaned_str != original_str:
                df.to_csv(csv_path, index=False)
                total_fixed += 1
                print(f"  Fixed: {csv_path.parent.name}/{csv_path.name}")
            else:
                total_unchanged += 1
        except Exception as e:
            print(f"  SKIP {csv_path.name}: {e}")

print(f"\n  Item 4 complete: {total_fixed} files fixed, {total_unchanged} unchanged.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Item 8 — Fix pathbuildB_metafamily_2d_mds.png colorbar: show 1,10,100,1000
# ══════════════════════════════════════════════════════════════════════════════
print("Item 8 — Regenerate pathbuildB_metafamily_2d_mds.png with proper log colorbar …")

summary = pd.read_csv(TABLE_DIR / "pathbuildB_metafamily_summary_consim1.csv")
families = pd.read_csv(TABLE_DIR / "pathbuildB_metafamilies_consim1.csv")

mf_name = dict(zip(summary["meta_family_id"], summary["dominant_family_name"]))
mf_paths = dict(zip(summary["meta_family_id"], summary["n_paths_total"]))
mf_n_fam = dict(zip(summary["meta_family_id"], summary["n_families"]))
mf_ids = sorted(summary["meta_family_id"].unique())

# Build meta-family Jaccard distances (same as rev4)
fam_sig = {}
for _, row in families.iterrows():
    sig = frozenset(
        c.strip() for c in str(row["signature_str"]).split(" & ") if ":" in c
    )
    fam_sig[row["family_id"]] = sig

mf_sigs = {}
for mf_id in mf_ids:
    all_fam = families[families["meta_family_id"] == mf_id]["family_id"].tolist()
    union = frozenset()
    for fid in all_fam:
        union = union | fam_sig.get(fid, frozenset())
    mf_sigs[mf_id] = union

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

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, max_iter=3000)
coords = mds.fit_transform(dist_mat)

path_counts = np.array([mf_paths.get(mid, 1) for mid in mf_ids])

# Use LogNorm so colormap is indexed by actual path counts
norm = LogNorm(vmin=max(path_counts.min(), 1), vmax=path_counts.max())
cmap = plt.cm.plasma

fig, ax = plt.subplots(figsize=(16, 12))
sc = ax.scatter(
    coords[:, 0],
    coords[:, 1],
    c=path_counts,
    cmap=cmap,
    norm=norm,
    s=[
        80 + 200 * (np.log10(max(pc, 1)) / np.log10(max(path_counts.max(), 1)))
        for pc in path_counts
    ],
    edgecolors="black",
    linewidths=0.5,
    zorder=3,
    alpha=0.85,
)

# Colorbar with actual path counts (1, 10, 100, 1000, 10000)
cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
cbar.set_label("N paths (log scale)", fontsize=10)
# Set ticks at powers of 10
tick_vals = [10**i for i in range(0, 6) if 10**i <= path_counts.max() * 1.5]
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([f"{v:,}" for v in tick_vals])

for i, mid in enumerate(mf_ids):
    name = mf_name.get(mid, f"MF{mid}")
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
    "Point size proportional to log(n_paths); color = n_paths (log scale)",
    fontsize=12,
)
plt.tight_layout()
out_mds = TABLE_DIR / "pathbuildB_metafamily_2d_mds.png"
fig.savefig(out_mds, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: step4_cluster_tables/{out_mds.name}")
print("  Item 8 complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Item 10 — Fix pathbuildB_metafamily_dendrogram.png: orientation="left"
#            so MF titles are horizontal and 1-Jaccard is the x-axis
# ══════════════════════════════════════════════════════════════════════════════
print(
    "Item 10 — Regenerate pathbuildB_metafamily_dendrogram.png (horizontal orientation) …"
)

Z = linkage(condensed, method="average")

# Load v2 chain names if available (will be used once D4 runs; falls back to summary names)
chain_names_v3_path = NAMING_DIR / "pathbuildB_chain_names_llm_v3.csv"
chain_names_v2_path = NAMING_DIR / "pathbuildB_chain_names_llm_v2.csv"

mf_llm_name = dict(mf_name)  # default to dominant_family_name from summary CSV

# Map v3 B-family names to meta-families via dominant_family_id in summary
# summary["dominant_family_id"] is a B-family rank (cluster_id in v3 CSV)
mf_dominant_fam = dict(zip(summary["meta_family_id"], summary["dominant_family_id"]))

if chain_names_v3_path.exists():
    cn = pd.read_csv(chain_names_v3_path)
    # v3 CSV uses cluster_id = B-family rank (0-39); map via dominant_family_id
    v3_names = dict(zip(cn["cluster_id"].astype(int), cn["final_name"]))
    for mf_id, dom_fam_id in mf_dominant_fam.items():
        v3_name = v3_names.get(int(dom_fam_id))
        if v3_name:
            mf_llm_name[mf_id] = v3_name
    print(
        f"  Using v3 chain names from {chain_names_v3_path.name} (mapped via dominant_family_id)"
    )
elif chain_names_v2_path.exists():
    cn = pd.read_csv(chain_names_v2_path)
    v2_names = dict(
        zip(cn["cluster_id"].astype(int), cn["final_name"].fillna(cn["llm_name"]))
    )
    for mf_id, dom_fam_id in mf_dominant_fam.items():
        v2_name = v2_names.get(int(dom_fam_id))
        if v2_name:
            mf_llm_name[mf_id] = v2_name
    print(f"  Using v2 chain names from {chain_names_v2_path.name}")
else:
    print("  (No chain names CSV found — using summary dominant_family_name)")

# Build labels: "MF{id}: {name}\n({n_paths:,} paths, {n_fam} fam)"
labels = []
for mid in mf_ids:
    name = mf_llm_name.get(mid, f"MF{mid}")
    n_p = mf_paths.get(mid, 0)
    n_f = mf_n_fam.get(mid, 0)
    labels.append(f"MF{mid}: {name}  ({n_p:,} paths, {n_f} fam)")

n_mf = len(mf_ids)
fig_h = max(12, n_mf * 0.42)
fig, ax = plt.subplots(figsize=(18, fig_h))

dend = dendrogram(
    Z,
    labels=labels,
    orientation="left",  # MF titles horizontal, metric on x-axis
    leaf_font_size=8,
    ax=ax,
    color_threshold=0.0,
    above_threshold_color="steelblue",
)

ax.set_xlabel("1 - Jaccard similarity (component union)", fontsize=11)
ax.set_title(
    "PathbuildB Meta-Family Dendrogram (k=32, average linkage on 1-Jaccard component similarity)\n"
    "Each meta-family represented by union of its constituent family component frozensets",
    fontsize=12,
)
plt.tight_layout()
out_dend = TABLE_DIR / "pathbuildB_metafamily_dendrogram.png"
fig.savefig(out_dend, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: step4_cluster_tables/{out_dend.name}")
print("  Item 10 complete.\n")

print("=== Rev5 fixes complete ===")
