"""
Phase 2 Step 4 — Three-layer network visualizations for consim0, consim1, consim2.

Uses PathbuildB (Option B frozenset co-occurrence families) as the L2 chain layer.
For each config, reads the connectivity CSVs and builds a three-column matplotlib figure:
  Left:   risk clusters    (R{id})
  Middle: B-family chains  (B{id})
  Right:  intervention clusters (I{id})

Edges width proportional to log(n_paths).
Node labels are full names (no truncation).

Outputs:
  step4_connectivity/three_layer_network_consim0.png
  step4_connectivity/three_layer_network_consim1.png
  step4_connectivity/three_layer_network_consim2.png
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "phase2_results")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4_finalanalysis")
OUT_CONN = os.path.join(STEP4_DIR, "step4_connectivity")
NAMING_DIR = os.path.join(RESULTS_DIR, "step5_naming")
LOG_DIR = os.path.join(ROOT, "logfiles", "phase4_logs")

os.makedirs(LOG_DIR, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, "phase4_three_layer_network.log")
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
log.info("Phase 2 Step 4 — Three-layer network visualizations")

# ─── File name map for each config (PathbuildB connectivity files) ────────────
CONFIG_FILES = {
    "consim0": {
        "r2c": "risk_to_Bfamily_edges_consim0.csv",
        "c2i": "Bfamily_to_interv_edges_consim0.csv",
        "r2i": "risk_to_interv_via_B_edges_consim0.csv",
        "out": "three_layer_network_pathbuildB_consim0.png",
    },
    "consim1": {
        "r2c": "risk_to_Bfamily_edges_consim1.csv",
        "c2i": "Bfamily_to_interv_edges_consim1.csv",
        "r2i": "risk_to_interv_via_B_edges_consim1.csv",
        "out": "three_layer_network_pathbuildB_consim1.png",
    },
    "consim2": {
        "r2c": "risk_to_Bfamily_edges_consim2.csv",
        "c2i": "Bfamily_to_interv_edges_consim2.csv",
        "r2i": "risk_to_interv_via_B_edges_consim2.csv",
        "out": "three_layer_network_pathbuildB_consim2.png",
    },
}


# ─── Load cluster name CSVs (best-effort, fall back to IDs) ──────────────────
def load_name_map(csv_path, id_col="cluster_id", name_col="final_name"):
    try:
        df = pd.read_csv(csv_path)
        # Fall back to llm_name if final_name is absent or all NaN
        if name_col not in df.columns or df[name_col].isna().all():
            name_col = "llm_name"
        return dict(zip(df[id_col].astype(str), df[name_col].fillna("")))
    except Exception:
        return {}


risk_names = load_name_map(os.path.join(NAMING_DIR, "risk_cluster_names_llm_v2.csv"))
interv_names = load_name_map(
    os.path.join(NAMING_DIR, "intervention_cluster_names_llm_v2.csv")
)
chain_names = load_name_map(
    os.path.join(NAMING_DIR, "pathbuildB_chain_names_llm_v2.csv")
)

log.info(
    f"  Loaded {len(risk_names)} risk names, {len(interv_names)} interv names, {len(chain_names)} chain names"
)


def make_three_layer_plot(config_name, r2c_df, c2i_df, r2i_df):
    """
    Build a three-column layout: Risk | Chain | Intervention.
    Nodes are positioned vertically in each column by frequency rank.
    Edge widths are proportional to log(n_paths+1).
    """
    # ── Select top nodes (by total n_paths) for each layer ──────────────────
    MAX_RISK = 20
    MAX_CHAIN = 20
    MAX_INTERV = 20
    MAX_EDGES = 150  # max edges to draw per pair of columns

    if len(r2i_df) > 0:
        top_risk = (
            r2i_df.groupby("cluster_a")["n_paths"]
            .sum()
            .nlargest(MAX_RISK)
            .index.tolist()
        )
        top_interv = (
            r2i_df.groupby("cluster_b")["n_paths"]
            .sum()
            .nlargest(MAX_INTERV)
            .index.tolist()
        )
    else:
        top_risk = []
        top_interv = []

    if len(r2c_df) > 0:
        top_chain = (
            r2c_df.groupby("cluster_b")["n_paths"]
            .sum()
            .nlargest(MAX_CHAIN)
            .index.tolist()
        )
    elif len(c2i_df) > 0:
        top_chain = (
            c2i_df.groupby("cluster_a")["n_paths"]
            .sum()
            .nlargest(MAX_CHAIN)
            .index.tolist()
        )
    else:
        top_chain = []

    # Convert all IDs to str for consistent lookup
    top_risk = [str(x) for x in top_risk]
    top_chain = [str(x) for x in top_chain]
    top_interv = [str(x) for x in top_interv]

    n_r = len(top_risk)
    n_c = len(top_chain)
    n_i = len(top_interv)
    if n_r == 0 and n_c == 0 and n_i == 0:
        log.warning(f"  [{config_name}] No data — skipping plot")
        return

    # ── Y positions (evenly spaced within column height) ─────────────────────
    def y_positions(items):
        n = len(items)
        if n == 0:
            return {}
        if n == 1:
            return {items[0]: 0.5}
        return {item: 1.0 - i / (n - 1) for i, item in enumerate(items)}

    risk_y = y_positions(top_risk)
    chain_y = y_positions(top_chain)
    interv_y = y_positions(top_interv)

    # X coordinates for the three columns
    X_RISK = 0.0
    X_CHAIN = 0.5
    X_INTERV = 1.0

    fig, ax = plt.subplots(figsize=(28, max(14, max(n_r, n_c, n_i) * 0.5)))

    # ── Draw edges: risk → chain ──────────────────────────────────────────────
    if len(r2c_df) > 0:
        sub = r2c_df[
            r2c_df["cluster_a"].astype(str).isin(top_risk)
            & r2c_df["cluster_b"].astype(str).isin(top_chain)
        ].head(MAX_EDGES)
        max_lp = np.log1p(sub["n_paths"].max()) if len(sub) > 0 else 1
        for _, row in sub.iterrows():
            rc = str(row["cluster_a"])
            cc = str(row["cluster_b"])
            ry = risk_y.get(rc)
            cy = chain_y.get(cc)
            if ry is None or cy is None:
                continue
            lw = max(0.2, np.log1p(row["n_paths"]) / max_lp * 4.0)
            ax.plot(
                [X_RISK, X_CHAIN], [ry, cy], color="steelblue", alpha=0.35, linewidth=lw
            )

    # ── Draw edges: chain → intervention ─────────────────────────────────────
    if len(c2i_df) > 0:
        sub = c2i_df[
            c2i_df["cluster_a"].astype(str).isin(top_chain)
            & c2i_df["cluster_b"].astype(str).isin(top_interv)
        ].head(MAX_EDGES)
        max_lp = np.log1p(sub["n_paths"].max()) if len(sub) > 0 else 1
        for _, row in sub.iterrows():
            cc = str(row["cluster_a"])
            ic = str(row["cluster_b"])
            cy = chain_y.get(cc)
            iy = interv_y.get(ic)
            if cy is None or iy is None:
                continue
            lw = max(0.2, np.log1p(row["n_paths"]) / max_lp * 4.0)
            ax.plot(
                [X_CHAIN, X_INTERV],
                [cy, iy],
                color="darkorange",
                alpha=0.35,
                linewidth=lw,
            )

    # ── Draw edges: risk → intervention (direct, dashed) ─────────────────────
    if len(r2i_df) > 0:
        sub = r2i_df[
            r2i_df["cluster_a"].astype(str).isin(top_risk)
            & r2i_df["cluster_b"].astype(str).isin(top_interv)
        ].head(MAX_EDGES)
        max_lp = np.log1p(sub["n_paths"].max()) if len(sub) > 0 else 1
        for _, row in sub.iterrows():
            rc = str(row["cluster_a"])
            ic = str(row["cluster_b"])
            ry = risk_y.get(rc)
            iy = interv_y.get(ic)
            if ry is None or iy is None:
                continue
            lw = max(0.2, np.log1p(row["n_paths"]) / max_lp * 2.5)
            ax.plot(
                [X_RISK, X_INTERV],
                [ry, iy],
                color="gray",
                alpha=0.15,
                linewidth=lw,
                linestyle="--",
                zorder=1,
            )

    # ── Draw nodes ────────────────────────────────────────────────────────────
    import textwrap

    def wrap(text, width=38):
        return "\n".join(textwrap.wrap(text, width))

    for cid, y in risk_y.items():
        ax.scatter(X_RISK, y, s=180, c="steelblue", zorder=4)
        label = wrap(risk_names.get(str(cid), f"R{cid}"))
        ax.text(
            X_RISK - 0.03,
            y,
            label,
            ha="right",
            va="center",
            fontsize=5.5,
            color="steelblue",
        )

    for cid, y in chain_y.items():
        ax.scatter(X_CHAIN, y, s=140, c="seagreen", zorder=4)
        label = wrap(chain_names.get(str(cid), f"B{cid}"), width=32)
        ax.text(
            X_CHAIN + 0.03,
            y,
            label,
            ha="left",
            va="center",
            fontsize=5,
            color="seagreen",
            rotation=0,
        )

    for cid, y in interv_y.items():
        ax.scatter(X_INTERV, y, s=180, c="darkorange", zorder=4)
        label = wrap(interv_names.get(str(cid), f"I{cid}"))
        ax.text(
            X_INTERV + 0.03,
            y,
            label,
            ha="left",
            va="center",
            fontsize=5.5,
            color="darkorange",
        )

    # ── Column headers ────────────────────────────────────────────────────────
    ax.text(
        X_RISK,
        1.05,
        "RISK\nClusters",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="steelblue",
    )
    ax.text(
        X_CHAIN,
        1.05,
        "CHAIN\n(PathbuildB Families)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="seagreen",
    )
    ax.text(
        X_INTERV,
        1.05,
        "INTERVENTION\nClusters",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="darkorange",
    )

    # ── Legend & titles ───────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="steelblue", label="Risk → Chain edges"),
        mpatches.Patch(color="darkorange", label="Chain → Interv edges"),
        mpatches.Patch(color="gray", label="Risk → Interv direct (dashed)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=8,
    )

    total_paths = int(r2i_df["n_paths"].sum()) if len(r2i_df) > 0 else 0
    ax.set_title(
        f"Three-Layer Network — {config_name}_pathbuildB\n"
        f"(top-{MAX_RISK} risk, top-{MAX_CHAIN} B-family chains, top-{MAX_INTERV} interv; "
        f"total r→i paths: {total_paths:,})",
        fontsize=11,
        pad=20,
    )
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(-0.15, 1.2)
    ax.axis("off")
    plt.tight_layout()

    out_path = os.path.join(OUT_CONN, CONFIG_FILES[config_name]["out"])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"  [{config_name}] Saved {out_path}")


# ─── Main loop over configs ───────────────────────────────────────────────────
for config_name, files in CONFIG_FILES.items():
    log.info("=" * 50)
    log.info(f"Processing {config_name} …")

    def load_csv(fname):
        fpath = os.path.join(OUT_CONN, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            # ensure cluster IDs are strings for consistent lookup
            for col in ["cluster_a", "cluster_b"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            log.info(f"  Loaded {fname}: {len(df)} rows")
            return df
        else:
            log.warning(f"  File not found: {fname} — using empty DataFrame")
            return pd.DataFrame(columns=["cluster_a", "cluster_b", "n_paths"])

    r2c_df = load_csv(files["r2c"])
    c2i_df = load_csv(files["c2i"])
    r2i_df = load_csv(files["r2i"])

    make_three_layer_plot(config_name, r2c_df, c2i_df, r2i_df)

log.info("=" * 70)
log.info("Three-layer network visualizations COMPLETE")
