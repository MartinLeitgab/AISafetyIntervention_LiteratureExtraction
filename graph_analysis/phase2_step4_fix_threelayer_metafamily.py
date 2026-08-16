"""
Fix and regenerate three_layer_network_pathbuildB_metafamily_consim1.png.

Problems fixed:
  1. head(MAX_EDGES=200) was cutting off all edges from lower-ranked risk clusters
     → removed, draw all filtered edges
  2. Meta-family labels had rotation=40 (invisible behind lines)
     → changed to rotation=0, positioned right of dot with ha="left"
  3. 20 meta-families had "B{id}" placeholder names (dominant family_id > 40,
     outside D3's top-40 naming scope)
     → generate names via GPT using decoded core_components

Outputs (in-place update + new PNG):
  step4_cluster_tables/pathbuildB_metafamily_summary_consim1.csv  (updated names)
  step4_connectivity/three_layer_network_pathbuildB_metafamily_consim1.png  (regenerated)
"""

import os
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).parent
load_dotenv(BASE / ".env", override=True)
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = os.environ["openai_api_key"]

TABLES_DIR = BASE / "phase2_results/step4_finalanalysis/step4_cluster_tables"
CONN_DIR = BASE / "phase2_results/step4_finalanalysis/step4_connectivity"
NAMING_DIR = BASE / "phase2_results/step5_naming"

MODEL = "gpt-5.4-mini"
client = OpenAI()

# ── 1. Load meta-family summary ───────────────────────────────────────────────
print("Loading meta-family summary …", flush=True)
meta_summary_df = pd.read_csv(TABLES_DIR / "pathbuildB_metafamily_summary_consim1.csv")
print(f"  {len(meta_summary_df)} meta-families")

# ── 2. Find unnamed meta-families (B{id} placeholders) ───────────────────────
unnamed_mask = meta_summary_df["dominant_family_name"].str.startswith("B", na=True)
unnamed_df = meta_summary_df[unnamed_mask].copy()
print(f"  {len(unnamed_df)} unnamed meta-families with placeholder names")

if len(unnamed_df) > 0:
    # Load component representative names
    print("Loading bodysubtype_cluster_representatives.csv …", flush=True)
    reps_df = pd.read_csv(TABLES_DIR / "bodysubtype_cluster_representatives.csv")
    rep_map = dict(zip(reps_df["prefix_key"], reps_df["rep_name"]))

    def decode_components(core_str: str) -> list[str]:
        """Convert 'de:15 & im:4 & pr:6' → list of human-readable descriptions."""
        if not isinstance(core_str, str) or not core_str.strip():
            return []
        names = []
        for part in core_str.split("&"):
            key = part.strip()
            name = rep_map.get(key, key)
            names.append(name)
        return names

    # ── 3. Generate names via GPT ─────────────────────────────────────────────
    print(
        f"\nGenerating names for {len(unnamed_df)} unnamed meta-families …", flush=True
    )
    SYSTEM_PROMPT = (
        "You name AI safety reasoning chain clusters. "
        "Each cluster groups R→C→I paths that share common reasoning components. "
        "You are given decoded descriptions of the core components in a cluster. "
        "Return a SHORT name (4-8 words) that describes the reasoning mechanism, "
        "in the style 'via [mechanism]' or 'through [concept1] and [concept2]'. "
        "Return ONLY the short name, no explanation."
    )

    new_names: dict[int, str] = {}
    for _, row in unnamed_df.iterrows():
        mf_id = int(row["meta_family_id"])
        core_comps = decode_components(str(row.get("core_components", "")))
        comp_text = (
            "\n".join(f"- {c}" for c in core_comps)
            if core_comps
            else "- (no core components)"
        )

        user_msg = (
            f"Meta-family MF{mf_id}: {int(row['n_families'])} B-families, "
            f"{int(row['n_paths_total'])} total paths.\n"
            f"Core reasoning components:\n{comp_text}\n\n"
            f"Provide a short mechanism name (4-8 words, starting with 'via' or 'through')."
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_completion_tokens=40,
            )
            name = resp.choices[0].message.content.strip().strip('"').strip("'")
            new_names[mf_id] = name
            print(f"  MF{mf_id}: {name}", flush=True)
        except Exception as e:
            print(f"  MF{mf_id}: ERROR {e}", flush=True)
            new_names[mf_id] = str(row["dominant_family_name"])
        time.sleep(0.15)  # mild rate limiting

    # ── 4. Update summary CSV ─────────────────────────────────────────────────
    for mf_id, name in new_names.items():
        meta_summary_df.loc[
            meta_summary_df["meta_family_id"] == mf_id, "dominant_family_name"
        ] = name

    meta_summary_df.to_csv(
        TABLES_DIR / "pathbuildB_metafamily_summary_consim1.csv", index=False
    )
    print(
        "\nUpdated summary saved → pathbuildB_metafamily_summary_consim1.csv",
        flush=True,
    )
else:
    print("  All meta-families already named — skipping GPT step.", flush=True)

# ── 5. Load edge files + cluster names ────────────────────────────────────────
print("\nLoading edge CSVs …", flush=True)
r2meta_df = pd.read_csv(CONN_DIR / "risk_to_metafamily_edges_consim1.csv")
meta2i_df = pd.read_csv(CONN_DIR / "metafamily_to_interv_edges_consim1.csv")
print(f"  r2meta: {len(r2meta_df)} edges, meta2i: {len(meta2i_df)} edges")


def load_names(path):
    try:
        df = pd.read_csv(path)
        col = "final_name" if "final_name" in df.columns else "llm_name"
        return {int(r["cluster_id"]): str(r[col]) for _, r in df.iterrows()}
    except Exception:
        return {}


risk_names = load_names(NAMING_DIR / "risk_cluster_names_llm_v2.csv")
interv_names = load_names(NAMING_DIR / "intervention_cluster_names_llm_v2.csv")
meta_names = {
    int(row["meta_family_id"]): row["dominant_family_name"]
    for _, row in meta_summary_df.iterrows()
}
meta_n_families = {
    int(row["meta_family_id"]): int(row["n_families"])
    for _, row in meta_summary_df.iterrows()
}

# ── 6. Select top nodes for display ──────────────────────────────────────────
MAX_RISK = 20
MAX_META = 32  # show all 32 meta-families
MAX_INTERV = 20

top_risk = (
    r2meta_df.groupby("risk_cluster")["n_paths"]
    .sum()
    .nlargest(MAX_RISK)
    .index.astype(int)
    .tolist()
)
top_meta = (
    r2meta_df.groupby("meta_family_id")["n_paths"]
    .sum()
    .nlargest(MAX_META)
    .index.astype(int)
    .tolist()
)
top_interv = (
    meta2i_df.groupby("interv_cluster")["n_paths"]
    .sum()
    .nlargest(MAX_INTERV)
    .index.astype(int)
    .tolist()
)

print(
    f"  top_risk: {len(top_risk)}, top_meta: {len(top_meta)}, top_interv: {len(top_interv)}",
    flush=True,
)


def y_positions(items):
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0]: 0.5}
    return {item: 1.0 - i / (n - 1) for i, item in enumerate(items)}


risk_y = y_positions(top_risk)
meta_y = y_positions(top_meta)
interv_y = y_positions(top_interv)

# Layout
X_RISK = 0.0
X_META = 0.5
X_INTERV = 1.0

n_r = len(top_risk)
n_m = len(top_meta)
n_i = len(top_interv)

# ── 7. Draw plot ──────────────────────────────────────────────────────────────
print("Drawing three-layer network …", flush=True)
fig_height = max(18, max(n_r, n_m, n_i) * 0.5)
fig, ax = plt.subplots(figsize=(36, fig_height))

# Risk → meta edges (ALL edges, no head() cutoff)
sub_r2m = r2meta_df[
    r2meta_df["risk_cluster"].astype(int).isin(top_risk)
    & r2meta_df["meta_family_id"].isin(top_meta)
]
print(f"  Drawing {len(sub_r2m)} risk→meta edges …", flush=True)
if len(sub_r2m) > 0:
    max_lp = np.log1p(sub_r2m["n_paths"].max())
    for _, row in sub_r2m.iterrows():
        rc = int(row["risk_cluster"])
        mc = int(row["meta_family_id"])
        ry = risk_y.get(rc)
        my = meta_y.get(mc)
        if ry is None or my is None:
            continue
        lw = max(0.15, np.log1p(row["n_paths"]) / max_lp * 3.5)
        ax.plot([X_RISK, X_META], [ry, my], color="steelblue", alpha=0.25, linewidth=lw)

# Meta → intervention edges (ALL edges, no head() cutoff)
sub_m2i = meta2i_df[
    meta2i_df["meta_family_id"].isin(top_meta)
    & meta2i_df["interv_cluster"].astype(int).isin(top_interv)
]
print(f"  Drawing {len(sub_m2i)} meta→interv edges …", flush=True)
if len(sub_m2i) > 0:
    max_lp = np.log1p(sub_m2i["n_paths"].max())
    for _, row in sub_m2i.iterrows():
        mc = int(row["meta_family_id"])
        ic = int(row["interv_cluster"])
        my = meta_y.get(mc)
        iy = interv_y.get(ic)
        if my is None or iy is None:
            continue
        lw = max(0.15, np.log1p(row["n_paths"]) / max_lp * 3.5)
        ax.plot(
            [X_META, X_INTERV], [my, iy], color="darkorange", alpha=0.25, linewidth=lw
        )


def wrap(text, width=32):
    return "\n".join(textwrap.wrap(str(text), width))


# Risk nodes (labels on left)
for cid, y in risk_y.items():
    ax.scatter(X_RISK, y, s=180, c="steelblue", zorder=5)
    ax.text(
        X_RISK - 0.03,
        y,
        wrap(risk_names.get(cid, f"R{cid}"), width=32),
        ha="right",
        va="center",
        fontsize=5.5,
        color="steelblue",
    )

# Meta-family nodes — FIX: rotation=0, labels to the RIGHT of dot
for mf_id, y in meta_y.items():
    n_fam_here = meta_n_families.get(mf_id, 1)
    ax.scatter(X_META, y, s=140, c="seagreen", zorder=5)
    label_text = (
        wrap(meta_names.get(mf_id, f"MF{mf_id}"), width=30) + f"\n({n_fam_here} fam)"
    )
    ax.text(
        X_META + 0.03,
        y,
        label_text,
        ha="left",
        va="center",
        fontsize=4.5,
        color="seagreen",
        rotation=0,
    )

# Intervention nodes (labels on right)
for cid, y in interv_y.items():
    ax.scatter(X_INTERV, y, s=180, c="darkorange", zorder=5)
    ax.text(
        X_INTERV + 0.03,
        y,
        wrap(interv_names.get(cid, f"I{cid}"), width=32),
        ha="left",
        va="center",
        fontsize=5.5,
        color="darkorange",
    )

# Headers
ax.text(
    X_RISK,
    1.07,
    "RISK\nClusters",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color="steelblue",
)
ax.text(
    X_META,
    1.07,
    f"META-CHAIN\n(PathbuildB Meta-Families, k={len(meta_y)})",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color="seagreen",
)
ax.text(
    X_INTERV,
    1.07,
    "INTERVENTION\nClusters",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color="darkorange",
)

legend_handles = [
    mpatches.Patch(color="steelblue", label="Risk → Meta-family"),
    mpatches.Patch(color="darkorange", label="Meta-family → Intervention"),
]
ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
    fontsize=9,
)

total_r2meta = int(r2meta_df["n_paths"].sum())
ax.set_title(
    f"Three-Layer Network — consim1_pathbuildB_metafamilies (k={len(meta_summary_df)})\n"
    f"(top-{MAX_RISK} risk, top-{MAX_META} meta-families, top-{MAX_INTERV} interv; "
    f"total r→meta paths: {total_r2meta:,}; "
    f"{len(sub_r2m)} R→meta edges, {len(sub_m2i)} meta→I edges shown)",
    fontsize=11,
    pad=20,
)
ax.set_xlim(-0.75, 1.75)
ax.set_ylim(-0.15, 1.25)
ax.axis("off")
plt.tight_layout()

out_png = CONN_DIR / "three_layer_network_pathbuildB_metafamily_consim1.png"
plt.savefig(out_png, dpi=130, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}", flush=True)
print("Done — three-layer network regenerated with all fixes.", flush=True)
