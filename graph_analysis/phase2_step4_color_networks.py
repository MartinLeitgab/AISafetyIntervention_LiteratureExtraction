"""
Phase 2 Step 4 — Color-coded three-layer network visualizations (Substep #27).

from collections import Counter
Produces per consim config:
  - three_layer_network_color_consimN.png  (Sankey-style layout)
  - three_layer_network_detail_consimN.png (labeled node-link, top-50 edges)
Also writes:
  - cluster_color_categories.csv
"""

from collections import Counter
import os
import pickle
import textwrap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/mnt/c/Users/malei/0_project_work/eleutherAI_SOAR_step1knowledgegraphcreation/AISafetyIntervention_LiteratureExtraction"
NAMING_DIR = f"{BASE}/graph_analysis/phase2_results/step5_naming"
TABLES_DIR = (
    f"{BASE}/graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables"
)
CONN_DIR = (
    f"{BASE}/graph_analysis/phase2_results/step4_finalanalysis/step4_connectivity"
)
PKL_DIR = f"{BASE}/graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
PATHS_FILE = (
    f"{BASE}/graph_analysis/phase1_rawpathsfiles/paths_unconstrained_sim0.9.jsonl"
)
OUT_DIR = CONN_DIR

# ---------------------------------------------------------------------------
# Color specs
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "x-risk": "#8B0000",
    "governance": "#6B238E",
    "near-term": "#FF8C00",
    "other": "#888888",
}
CHAIN_COLORS = {
    "technical alignment": "#1565C0",
    "oversight & governance": "#2E7D32",
    "near-term safety": "#F9A825",
    "field-building": "#00695C",
    "other": "#888888",
}
INTERV_COLORS = {
    "design lifecycle": "#90CAF9",
    "training lifecycle": "#1976D2",
    "deployment lifecycle": "#0D47A1",
}

# ---------------------------------------------------------------------------
# Classification keyword lists
# ---------------------------------------------------------------------------
XRISK_KEYWORDS = [
    "existential",
    "extinction",
    "catastroph",
    "misalign",
    "agi",
    "superintelligent",
    "disempowerment",
    "power-seeking",
    "recursive self-improvement",
    "runaway",
    "advanced ai value",
    "takeover",
    "global catastrophic",
]
GOVFAIL_KEYWORDS = [
    "arms race",
    "oversight",
    "opaque",
    "forecasting",
    "insufficient",
    "shutdown",
    "job displacement",
    "concentration of wealth",
    "societal harm",
    "bias-driven",
    "unreliable ai decisions",
    "recommender",
    "polarize",
    "timeline",
]
NEARTERM_KEYWORDS = [
    "out-of-distribution",
    "ood",
    "reward misspec",
    "adversarial",
    "deceptive alignment",
    "harmful",
    "misleading",
    "discrimination",
    "undetected",
    "privacy",
    "coordination failure",
    "sample inefficiency",
    "reward hack",
]

FIELDBUILD_KEYWORDS = [
    "field-building",
    "research capacity",
    "funding",
    "education",
    "outreach",
    "cost-effectiveness",
    "talent",
]
NEARTERMSAFETY_KEYWORDS = [
    "out-of-distribution",
    "adversarial example",
    "robustness",
    "generalization failure",
]
OVERSIGHT_KEYWORDS = [
    "governance",
    "oversight",
    "audit",
    "monitoring",
    "accountability",
    "policy",
    "regulation",
    "safety review",
    "deployment",
    "inference-time",
]
TECHALGN_KEYWORDS = [
    "rl",
    "reward",
    "rlhf",
    "adversarial",
    "generalization",
    "reinforcement",
    "specification",
    "objective",
    "ood",
    "robustness",
    "training",
    "fine-tuning",
    "alignment failure",
    "misspecification",
    "inner objective",
]


def classify_risk(name: str) -> str:
    n = name.lower()
    for kw in XRISK_KEYWORDS:
        if kw in n:
            return "x-risk"
    for kw in GOVFAIL_KEYWORDS:
        if kw in n:
            return "governance"
    for kw in NEARTERM_KEYWORDS:
        if kw in n:
            return "near-term"
    return "other"


def classify_chain(name: str) -> str:
    n = name.lower()
    for kw in FIELDBUILD_KEYWORDS:
        if kw in n:
            return "field-building"
    for kw in NEARTERMSAFETY_KEYWORDS:
        if kw in n:
            return "near-term safety"
    for kw in OVERSIGHT_KEYWORDS:
        if kw in n:
            return "oversight & governance"
    for kw in TECHALGN_KEYWORDS:
        if kw in n:
            return "technical alignment"
    return "other"


# ---------------------------------------------------------------------------
# Load naming CSV → {cluster_id: final_name}
# ---------------------------------------------------------------------------
def load_names(path: str) -> dict:
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        cid = int(row["cluster_id"])
        name = (
            str(row["final_name"])
            if pd.notna(row["final_name"])
            else str(row["llm_name"])
        )
        result[cid] = name
    return result


# ---------------------------------------------------------------------------
# Load cluster sizing data
# ---------------------------------------------------------------------------
def load_risk_sizes(path: str) -> dict:
    """Returns {cluster_id: n_nodes}"""
    df = pd.read_csv(path)
    col = "n_nodes" if "n_nodes" in df.columns else "n_qualifying_nodes"
    return {int(r["cluster_id"]): int(r[col]) for _, r in df.iterrows()}


def load_interv_sizes(path: str) -> dict:
    df = pd.read_csv(path)
    col = "n_nodes" if "n_nodes" in df.columns else "n_qualifying_nodes"
    return {int(r["cluster_id"]): int(r[col]) for _, r in df.iterrows()}


def load_chain_sizes(path: str) -> dict:
    """Chain clusters sized by n_paths"""
    df = pd.read_csv(path)
    return {int(r["cluster_id"]): int(r["n_paths"]) for _, r in df.iterrows()}


# ---------------------------------------------------------------------------
# Classify intervention clusters by mean lifecycle
# ---------------------------------------------------------------------------
def compute_interv_lifecycle(
    node_attrs: dict,
    cluster_memberships: dict,
    valid_nodes: set,
    edge_config: str,
    mode: str,
    node_type: str,
    algo: str,
) -> dict:
    """Returns {cluster_id: mean_lifecycle}"""
    lifecycle_by_cluster = {}
    for key, members in cluster_memberships.items():
        if len(key) >= 5:
            ec, m, nt, alg, cid = key[0], key[1], key[2], key[3], key[4]
        else:
            continue
        if ec != edge_config or m != mode or nt != node_type or alg != algo:
            continue
        vals = []
        for nid in members:
            if valid_nodes and nid not in valid_nodes:
                continue
            attrs = node_attrs.get(nid, {})
            lc = attrs.get("intervention_lifecycle")
            if lc is not None:
                try:
                    vals.append(float(lc))
                except (ValueError, TypeError):
                    pass
        if vals:
            lifecycle_by_cluster[int(cid)] = float(np.mean(vals))
    return lifecycle_by_cluster


def classify_interv_lifecycle(mean_lc: float) -> str:
    if mean_lc <= 2.5:
        return "design lifecycle"
    elif mean_lc <= 4.5:
        return "training lifecycle"
    else:
        return "deployment lifecycle"


# ---------------------------------------------------------------------------
# Load valid pathway nodes from JSONL
# ---------------------------------------------------------------------------
def load_valid_pathway_nodes(path: str) -> set:
    import json

    nodes = set()
    if not os.path.exists(path):
        print(f"  WARNING: paths file not found: {path}")
        return nodes
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Each path is a list of node IDs; collect all
                if isinstance(obj, list):
                    for nid in obj:
                        nodes.add(nid)
                elif isinstance(obj, dict):
                    # try common fields
                    for field in ("path", "nodes", "node_ids"):
                        if field in obj:
                            for nid in obj[field]:
                                nodes.add(nid)
                            break
            except Exception:
                pass
    print(f"  Loaded {len(nodes)} valid pathway nodes from JSONL")
    return nodes


# ---------------------------------------------------------------------------
# Load PKL files (heavy — done once)
# ---------------------------------------------------------------------------
print("Loading PKL files...")
with open(f"{PKL_DIR}/graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
print(f"  node_attrs: {len(node_attrs)} nodes")

with open(f"{PKL_DIR}/cluster_memberships.pkl", "rb") as f:
    cluster_memberships = pickle.load(f)
print(f"  cluster_memberships: {len(cluster_memberships)} keys")

# Load valid pathway nodes
valid_pathway_nodes = load_valid_pathway_nodes(PATHS_FILE)

# ---------------------------------------------------------------------------
# Load naming data
# ---------------------------------------------------------------------------
risk_names = load_names(f"{NAMING_DIR}/risk_cluster_names_llm.csv")
chain_names = load_names(f"{NAMING_DIR}/chain_cluster_names_llm.csv")
interv_names = load_names(f"{NAMING_DIR}/intervention_cluster_names_llm.csv")

# ---------------------------------------------------------------------------
# Classify risk and chain clusters (same for all configs, based on naming CSV)
# ---------------------------------------------------------------------------
risk_categories = {cid: classify_risk(name) for cid, name in risk_names.items()}
chain_categories = {cid: classify_chain(name) for cid, name in chain_names.items()}

print("\nRisk cluster categories:")
for cid in sorted(risk_categories.keys()):
    print(f"  {cid:3d}  {risk_categories[cid]:15s}  {risk_names[cid][:60]}")

print("\nChain cluster categories:")
for cid in sorted(chain_categories.keys()):
    print(f"  {cid:3d}  {chain_categories[cid]:25s}  {chain_names[cid][:55]}")

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
# consim0: edge_config='edge_only', mode='risk',    node_type='intervention'
# consim1: edge_config='sim09',     mode='both',    node_type='intervention'
# consim2: edge_config='sim09',     mode='both',    node_type='intervention'  (unconstrained paths)
#
# For lifecycle computation we need the actual cluster_memberships key structure.
# Let's inspect what keys exist to find the right ones.
sample_keys = list(cluster_memberships.keys())[:10]
print(f"\nSample cluster_memberships keys: {sample_keys}")

# Discover unique (edge_config, mode, node_type, algo) combos
combos = set()
for key in cluster_memberships.keys():
    if len(key) >= 5:
        combos.add((str(key[0]), key[1], key[2], key[3]))
print(f"Unique (edge_config, mode, node_type, algo) combos ({len(combos)} total):")
for c in sorted(combos):
    print(f"  {c}")

# ---------------------------------------------------------------------------
# Intervention lifecycle per config
# We'll pick 'agglomerative' algo, node_type='intervention'
# ---------------------------------------------------------------------------
# Identify algo name used
algo_options = set(k[3] for k in cluster_memberships.keys() if len(k) >= 5)
print(f"\nAlgo options: {algo_options}")
ALGO = "agglomerative" if "agglomerative" in algo_options else list(algo_options)[0]
print(f"Using algo: {ALGO}")

# Consim configs: map config label -> (edge_config_as_in_pkl, mode)
# consim0: edge-only paths  -> ('EDGE', 'unconstrained', 'intervention', 'agglomerative')
# consim1: sim0.9 paths     -> (0.9, 'unconstrained', 'intervention', 'agglomerative')
# consim2: sim0.9 unconstrained (same clustering as consim1, different path file)
CONSIM_CONFIGS = {
    "consim0": ("EDGE", "unconstrained"),
    "consim1": (0.9, "unconstrained"),
    "consim2": (0.9, "unconstrained"),  # same cluster assignments as consim1
}


def compute_interv_lifecycle_v2(
    node_attrs: dict,
    cluster_memberships: dict,
    ec_key,
    mode: str,
    algo: str = "agglomerative",
) -> dict:
    """Returns {cluster_id: mean_lifecycle} using exact key matching (float or str)."""
    lifecycle_by_cluster = {}
    for key, members in cluster_memberships.items():
        if len(key) < 5:
            continue
        k_ec, k_mode, k_nt, k_algo, k_cid = key[0], key[1], key[2], key[3], key[4]
        # Match edge_config
        if isinstance(ec_key, float):
            try:
                ec_match = abs(float(k_ec) - ec_key) < 1e-9
            except (ValueError, TypeError):
                ec_match = False
        else:
            ec_match = str(k_ec) == str(ec_key)
        if not (
            ec_match and k_mode == mode and k_nt == "intervention" and k_algo == algo
        ):
            continue
        cid = int(k_cid)
        vals = []
        for nid in members:
            attrs = node_attrs.get(nid, {})
            lc_val = attrs.get("intervention_lifecycle")
            if lc_val is not None:
                try:
                    vals.append(float(lc_val))
                except (ValueError, TypeError):
                    pass
        if vals:
            lifecycle_by_cluster[cid] = float(np.mean(vals))
    return lifecycle_by_cluster


interv_lifecycle_by_config = {}
for config_label, (ec_key, mode) in CONSIM_CONFIGS.items():
    print(
        f"\nComputing intervention lifecycle for {config_label} (ec={ec_key}, mode={mode})..."
    )
    lc = compute_interv_lifecycle_v2(
        node_attrs, cluster_memberships, ec_key, mode, ALGO
    )
    print(f"  Found lifecycle data for {len(lc)} intervention clusters")
    interv_lifecycle_by_config[config_label] = lc

# Classify intervention clusters per config
interv_categories_by_config = {}
for config_label, lc_map in interv_lifecycle_by_config.items():
    cats = {}
    for cid in interv_names.keys():
        if cid in lc_map:
            cats[cid] = classify_interv_lifecycle(lc_map[cid])
        else:
            cats[cid] = "training lifecycle"  # default
    interv_categories_by_config[config_label] = cats

# ---------------------------------------------------------------------------
# Build master category table for CSV export
# ---------------------------------------------------------------------------
rows = []
for cid, name in risk_names.items():
    rows.append(
        {
            "layer": "L1_risk",
            "cluster_id": cid,
            "final_name": name,
            "category": risk_categories[cid],
        }
    )
for cid, name in chain_names.items():
    rows.append(
        {
            "layer": "L2_chain",
            "cluster_id": cid,
            "final_name": name,
            "category": chain_categories[cid],
        }
    )
# Use consim1 as representative for intervention categories
interv_cats_rep = interv_categories_by_config.get(
    "consim1", interv_categories_by_config.get("consim0", {})
)
for cid, name in interv_names.items():
    rows.append(
        {
            "layer": "L3_intervention",
            "cluster_id": cid,
            "final_name": name,
            "category": interv_cats_rep.get(cid, "training lifecycle"),
        }
    )
cat_df = pd.DataFrame(rows)
cat_df.to_csv(f"{OUT_DIR}/cluster_color_categories.csv", index=False)
print(f"\nSaved cluster_color_categories.csv ({len(cat_df)} rows)")

# ---------------------------------------------------------------------------
# Connectivity file paths per config
# ---------------------------------------------------------------------------
CONN_FILES = {
    "consim0": {
        "rc": f"{CONN_DIR}/risk_to_chain_edges_consim0.csv",
        "ci": f"{CONN_DIR}/chain_to_interv_edges_consim0.csv",
        "ri": f"{CONN_DIR}/risk_to_interv_edges_consim0.csv",
        "risk_tbl": f"{TABLES_DIR}/risk_clusters_consim0.csv",
        "interv_tbl": f"{TABLES_DIR}/intervention_clusters_consim0.csv",
        "chain_tbl": f"{TABLES_DIR}/optionA_chainbody_clusters_consim0.csv",
    },
    "consim1": {
        "rc": f"{CONN_DIR}/risk_to_chain_edges_consim1.csv",
        "ci": f"{CONN_DIR}/chain_to_interv_edges_consim1.csv",
        "ri": None,  # check if exists
        "risk_tbl": f"{TABLES_DIR}/risk_clusters_consim1.csv",
        "interv_tbl": f"{TABLES_DIR}/intervention_clusters_consim1.csv",
        "chain_tbl": f"{TABLES_DIR}/optionA_chainbody_clusters_consim1.csv",
    },
    "consim2": {
        "rc": f"{CONN_DIR}/risk_to_chain_edges.csv",
        "ci": f"{CONN_DIR}/chain_to_intervention_edges.csv",
        "ri": f"{CONN_DIR}/risk_to_intervention_edges.csv",
        "risk_tbl": f"{TABLES_DIR}/risk_clusters.csv",
        "interv_tbl": f"{TABLES_DIR}/intervention_clusters.csv",
        "chain_tbl": f"{TABLES_DIR}/optionA_chainbody_clusters.csv",
    },
}

# Check if consim1 has ri file
ri1 = f"{CONN_DIR}/risk_to_interv_edges_consim1.csv"
CONN_FILES["consim1"]["ri"] = ri1 if os.path.exists(ri1) else None


# ---------------------------------------------------------------------------
# Utility: scale node sizes
# ---------------------------------------------------------------------------
def scale_size(
    n: float, min_s: float = 50, max_s: float = 500, all_vals: list = None
) -> float:
    if all_vals is None or len(all_vals) <= 1:
        return 200
    mn, mx = min(all_vals), max(all_vals)
    if mx == mn:
        return (min_s + max_s) / 2
    frac = (n - mn) / (mx - mn)
    return min_s + frac * (max_s - min_s)


def scale_edge_width(
    n_paths: float, max_paths: float, min_w: float = 0.1, max_w: float = 2.0
) -> float:
    if max_paths <= 0:
        return min_w
    return min_w + (np.log1p(n_paths) / np.log1p(max_paths)) * (max_w - min_w)


def scale_edge_alpha(
    n_paths: float, max_paths: float, min_a: float = 0.05, max_a: float = 0.6
) -> float:
    if max_paths <= 0:
        return min_a
    return min_a + (np.log1p(n_paths) / np.log1p(max_paths)) * (max_a - min_a)


def short_name(name: str, n_words: int = 3) -> str:
    words = name.split()
    return " ".join(words[:n_words])


# ---------------------------------------------------------------------------
# MAIN PLOTTING FUNCTION
# ---------------------------------------------------------------------------
def make_plots(config_label: str, files: dict, interv_cats: dict):
    print(f"\n{'=' * 60}")
    print(f"  Processing {config_label}")
    print(f"{'=' * 60}")

    # --- Load cluster tables ---
    risk_sizes = load_risk_sizes(files["risk_tbl"])
    interv_sizes = load_interv_sizes(files["interv_tbl"])
    chain_sizes = load_chain_sizes(files["chain_tbl"])

    print(
        f"  Risk clusters: {len(risk_sizes)}, Chain: {len(chain_sizes)}, Interv: {len(interv_sizes)}"
    )

    # --- Load connectivity edges ---
    rc_df = pd.read_csv(files["rc"])  # risk→chain
    ci_df = pd.read_csv(files["ci"])  # chain→interv
    ri_df = (
        pd.read_csv(files["ri"])
        if files["ri"] and os.path.exists(files["ri"])
        else pd.DataFrame(columns=["cluster_a", "cluster_b", "n_paths"])
    )

    print(
        f"  R→C edges: {len(rc_df)}, C→I edges: {len(ci_df)}, R→I edges: {len(ri_df)}"
    )

    # --- Build cluster lists (sorted by cluster_id) ---
    risk_cids = sorted(risk_sizes.keys())
    chain_cids = sorted(chain_sizes.keys())
    interv_cids = sorted(interv_sizes.keys())

    # --- Assign y positions ---
    def y_positions(cids):
        n = len(cids)
        return {cid: (i / max(n - 1, 1)) for i, cid in enumerate(cids)}

    risk_y = y_positions(risk_cids)
    chain_y = y_positions(chain_cids)
    interv_y = y_positions(interv_cids)

    # ========================================================
    # PLOT 1: Sankey-style color network
    # ========================================================
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    title = (
        f"Three-Layer Network — {config_label}  "
        f"({len(risk_cids)}R × {len(chain_cids)}C × {len(interv_cids)}I clusters)"
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Column headers
    for x, lbl in [(0, "L1 Risk"), (1, "L2 Chain"), (2, "L3 Intervention")]:
        ax.text(
            x,
            1.04,
            lbl,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#222222",
        )

    # --- Draw edges FIRST (behind nodes) ---
    all_paths = (
        list(rc_df["n_paths"]) + list(ci_df["n_paths"]) + list(ri_df["n_paths"])
        if len(ri_df)
        else list(rc_df["n_paths"]) + list(ci_df["n_paths"])
    )
    max_paths = max(all_paths) if all_paths else 1

    # R→C edges
    for _, row in rc_df.iterrows():
        ra, cb = int(row["cluster_a"]), int(row["cluster_b"])
        np_ = float(row["n_paths"])
        if ra not in risk_y or cb not in chain_y:
            continue
        w = scale_edge_width(np_, max_paths)
        a = scale_edge_alpha(np_, max_paths)
        ax.plot(
            [0, 1], [risk_y[ra], chain_y[cb]], color="#999999", lw=w, alpha=a, zorder=1
        )

    # C→I edges
    for _, row in ci_df.iterrows():
        ca, ib = int(row["cluster_a"]), int(row["cluster_b"])
        np_ = float(row["n_paths"])
        if ca not in chain_y or ib not in interv_y:
            continue
        w = scale_edge_width(np_, max_paths)
        a = scale_edge_alpha(np_, max_paths)
        ax.plot(
            [1, 2],
            [chain_y[ca], interv_y[ib]],
            color="#999999",
            lw=w,
            alpha=a,
            zorder=1,
        )

    # --- Draw nodes ---
    all_risk_sizes = list(risk_sizes.values())
    all_chain_sizes = list(chain_sizes.values())
    all_interv_sizes = list(interv_sizes.values())

    # Risk nodes
    for cid in risk_cids:
        y = risk_y[cid]
        cat = risk_categories.get(cid, "other")
        color = RISK_COLORS[cat]
        sz = scale_size(risk_sizes[cid], all_vals=all_risk_sizes)
        ax.scatter(0, y, s=sz, c=color, zorder=3, edgecolors="white", linewidths=0.5)
        name_s = short_name(risk_names.get(cid, f"R{cid}"))
        ax.text(
            0 - 0.04,
            y,
            f"{cid}: {name_s}",
            ha="right",
            va="center",
            fontsize=5.5,
            color="#333333",
        )

    # Chain nodes
    for cid in chain_cids:
        y = chain_y[cid]
        cat = chain_categories.get(cid, "other")
        color = CHAIN_COLORS[cat]
        sz = scale_size(chain_sizes[cid], all_vals=all_chain_sizes)
        ax.scatter(1, y, s=sz, c=color, zorder=3, edgecolors="white", linewidths=0.5)
        name_s = short_name(chain_names.get(cid, f"C{cid}"))
        ax.text(
            1 + 0.04,
            y,
            f"{cid}: {name_s}",
            ha="left",
            va="center",
            fontsize=5.5,
            color="#333333",
        )

    # Intervention nodes
    for cid in interv_cids:
        y = interv_y[cid]
        cat = interv_cats.get(cid, "training lifecycle")
        color = INTERV_COLORS[cat]
        sz = scale_size(interv_sizes[cid], all_vals=all_interv_sizes)
        ax.scatter(2, y, s=sz, c=color, zorder=3, edgecolors="white", linewidths=0.5)
        name_s = short_name(interv_names.get(cid, f"I{cid}"))
        ax.text(
            2 + 0.04,
            y,
            f"{cid}: {name_s}",
            ha="left",
            va="center",
            fontsize=5.5,
            color="#333333",
        )

    # --- Legend ---
    legend_handles = []
    for cat, col in RISK_COLORS.items():
        legend_handles.append(mpatches.Patch(color=col, label=f"Risk: {cat}"))
    for cat, col in CHAIN_COLORS.items():
        legend_handles.append(mpatches.Patch(color=col, label=f"Chain: {cat}"))
    for cat, col in INTERV_COLORS.items():
        legend_handles.append(mpatches.Patch(color=col, label=f"Interv: {cat}"))
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=7,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    out1 = f"{OUT_DIR}/three_layer_network_color_{config_label}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out1}")

    # ========================================================
    # PLOT 2: Detail plot — top-50 edges, full labels
    # ========================================================
    fig2, ax2 = plt.subplots(figsize=(24, 18))
    ax2.set_xlim(-0.4, 2.4)
    ax2.set_ylim(-0.05, 1.08)
    ax2.axis("off")
    ax2.set_title(
        title + " [Detail — top 50 edges]", fontsize=14, fontweight="bold", pad=12
    )

    for x, lbl in [(0, "L1 Risk"), (1, "L2 Chain"), (2, "L3 Intervention")]:
        ax2.text(
            x,
            1.06,
            lbl,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#222222",
        )

    # Combine all edges, pick top 50 by n_paths
    rc_df2 = rc_df.copy()
    rc_df2["from_layer"] = "risk"
    rc_df2["to_layer"] = "chain"

    ci_df2 = ci_df.copy()
    ci_df2["from_layer"] = "chain"
    ci_df2["to_layer"] = "interv"

    all_edges = pd.concat([rc_df2, ci_df2], ignore_index=True)
    all_edges = all_edges.sort_values("n_paths", ascending=False).head(50)

    max_paths_top50 = float(all_edges["n_paths"].max()) if len(all_edges) else 1

    for _, row in all_edges.iterrows():
        ca, cb = int(row["cluster_a"]), int(row["cluster_b"])
        np_ = float(row["n_paths"])
        from_layer = row["from_layer"]
        row["to_layer"]

        if from_layer == "risk":
            if ca not in risk_y or cb not in chain_y:
                continue
            x0, y0 = 0, risk_y[ca]
            x1, y1 = 1, chain_y[cb]
            src_cat = risk_categories.get(ca, "other")
            src_color = RISK_COLORS[src_cat]
        else:
            if ca not in chain_y or cb not in interv_y:
                continue
            x0, y0 = 1, chain_y[ca]
            x1, y1 = 2, interv_y[cb]
            src_cat = chain_categories.get(ca, "other")
            src_color = CHAIN_COLORS[src_cat]

        lw = 0.3 + (np.log1p(np_) / np.log1p(max_paths_top50)) * 5.7
        lw = min(lw, 6.0)

        import matplotlib.colors as mcolors

        r, g, b, _ = mcolors.to_rgba(src_color)
        edge_color = (r, g, b, 0.5)
        ax2.plot([x0, x1], [y0, y1], color=edge_color, lw=lw, zorder=1)

    # Draw ALL nodes (even those with no top-50 edges)
    for cid in risk_cids:
        y = risk_y[cid]
        cat = risk_categories.get(cid, "other")
        color = RISK_COLORS[cat]
        sz = scale_size(risk_sizes[cid], 60, 600, all_vals=all_risk_sizes)
        ax2.scatter(0, y, s=sz, c=color, zorder=3, edgecolors="white", linewidths=0.5)
        full_name = risk_names.get(cid, f"R{cid}")
        wrapped = "\n".join(textwrap.wrap(full_name, 25))
        ax2.text(
            0 - 0.05,
            y,
            f"{cid}: {wrapped}",
            ha="right",
            va="center",
            fontsize=5,
            color="#222222",
            linespacing=1.2,
        )

    for cid in chain_cids:
        y = chain_y[cid]
        cat = chain_categories.get(cid, "other")
        color = CHAIN_COLORS[cat]
        sz = scale_size(chain_sizes[cid], 60, 600, all_vals=all_chain_sizes)
        ax2.scatter(1, y, s=sz, c=color, zorder=3, edgecolors="white", linewidths=0.5)
        full_name = chain_names.get(cid, f"C{cid}")
        wrapped = "\n".join(textwrap.wrap(full_name, 25))
        ax2.text(
            1 + 0.05,
            y,
            f"{cid}: {wrapped}",
            ha="left",
            va="center",
            fontsize=5,
            color="#222222",
            linespacing=1.2,
        )

    for cid in interv_cids:
        y = interv_y[cid]
        cat = interv_cats.get(cid, "training lifecycle")
        color = INTERV_COLORS[cat]
        sz = scale_size(interv_sizes[cid], 60, 600, all_vals=all_interv_sizes)
        ax2.scatter(2, y, s=sz, c=color, zorder=3, edgecolors="white", linewidths=0.5)
        full_name = interv_names.get(cid, f"I{cid}")
        wrapped = "\n".join(textwrap.wrap(full_name, 25))
        ax2.text(
            2 + 0.05,
            y,
            f"{cid}: {wrapped}",
            ha="left",
            va="center",
            fontsize=5,
            color="#222222",
            linespacing=1.2,
        )

    ax2.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=7,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    out2 = f"{OUT_DIR}/three_layer_network_detail_{config_label}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  Saved: {out2}")


# ---------------------------------------------------------------------------
# Run for all 3 configs
# ---------------------------------------------------------------------------
for config_label in ["consim0", "consim1", "consim2"]:
    files = CONN_FILES[config_label]
    interv_cats = interv_categories_by_config.get(config_label, {})
    make_plots(config_label, files, interv_cats)

print("\nAll done.")

# ---------------------------------------------------------------------------
# Print category summary
# ---------------------------------------------------------------------------
print("\n=== RISK category breakdown ===")

print(Counter(risk_categories.values()))

print("\n=== CHAIN category breakdown ===")
print(Counter(chain_categories.values()))

print("\n=== INTERVENTION category breakdown (consim1) ===")
cats_c1 = interv_categories_by_config.get("consim1", {})
print(Counter(cats_c1.values()))
