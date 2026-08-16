"""
Reproducible calculations for Phase2_Step2_Issues.md items 2, 4, 5, 7, 8.

Outputs a JSON results file: phase2_results/reproducible_calculations.json
Also prints a human-readable summary to stdout (redirect to log).

Issue 2: Count SIM edges where BOTH endpoints are in valid-pathway node set,
         at each threshold — to add as a column to the SIMILARITY table.

Issue 4: Recompute the "107×" claim at a consistent threshold.
         Correct: % of migrating nodes with centroid_sim > T vs % of random
         cluster pairs with centroid_sim > T for the same T.

Issue 5: Compute inter-cluster similarity baseline from the 40×40 centroid
         similarity matrix for EDGE/unconstrained/risk config.
         Outputs: mean off-diagonal sim, range, % pairs > 0.9.

Issue 7: Pairwise cosine similarities between top-5 hub nodes (from
         hub_quality_metrics.csv, post-fix) using graph_node_attributes.pkl.

Issue 8: Count paths below the ≥5-hop minimum across all paths_*.jsonl files.
         Outputs: total paths, paths < 5 hops, percentage removed.

Run from graph_analysis/:
    python phase2_reproducible_calculations.py > /tmp/reproducible_calc.log 2>&1
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

STEP1_DIR = Path("phase2_results/step1_load_and_parse_umapwithoutlocalsatellites")
STEP2_DIR = Path("phase2_results/step2_metrics_and_stability")
PATHS_DIR = Path("phase1_rawpathsfiles")
OUT_FILE = Path("phase2_results/reproducible_calculations.json")

results = {}


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


# ─── LOAD VALID-PATHWAY NODE SET ─────────────────────────────────────────────
print("=" * 70)
print("Building valid-pathway node set …")
print("=" * 70)
valid_nodes: set = set()
for pf in sorted(PATHS_DIR.glob("paths_*.jsonl")):
    n_before = len(valid_nodes)
    with open(pf) as f:
        for line in f:
            rec = json.loads(line)
            path = rec.get("path", [])
            if isinstance(path, str):
                path = json.loads(path)
            valid_nodes.update(str(n) for n in path)
print(f"  Total valid-pathway nodes: {len(valid_nodes):,}")


# ─── LOAD PKLs ────────────────────────────────────────────────────────────────
print("\nLoading node_attrs …")
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
print(f"  {len(node_attrs):,} nodes")

print("Loading edge_data …")
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)
print(f"  {len(edge_data):,} edges")

print("Loading cluster_memberships …")
with open(STEP1_DIR / "cluster_memberships.pkl", "rb") as f:
    cluster_memberships = pickle.load(f)
print(f"  {len(cluster_memberships):,} cluster records")


# ─── ISSUE 2: Valid-pathway SIM edge counts ───────────────────────────────────
print("\n" + "=" * 70)
print("Issue 2: Valid-pathway SIM edge counts at each threshold")
print("=" * 70)

THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
full_counts = defaultdict(int)
valid_counts = defaultdict(int)

for e in edge_data:
    if str(e.get("type", "")).upper() != "SIMILARITY":
        continue
    score = e.get("similarity_score")
    if score is None:
        continue
    cos_sim = cos_sim_from_score(score)
    src = str(e.get("source", ""))
    tgt = str(e.get("target", ""))
    both_valid = src in valid_nodes and tgt in valid_nodes

    for thr in THRESHOLDS:
        if cos_sim >= thr:
            full_counts[thr] += 1
            if both_valid:
                valid_counts[thr] += 1

print(f"\n{'Threshold':<12} {'Full PKL':>12} {'Valid-pathway':>14} {'Valid %':>9}")
print("-" * 52)
issue2 = {}
for thr in THRESHOLDS:
    fc = full_counts[thr]
    vc = valid_counts[thr]
    pct = 100 * vc / fc if fc > 0 else 0
    label = f"{thr:.2f}"
    print(f"  ≥ {label}    {fc:>12,} {vc:>14,} {pct:>8.1f}%")
    issue2[label] = {"full_pkL": fc, "valid_pathway": vc, "valid_pct": round(pct, 1)}
results["issue2_sim_edge_counts"] = issue2


# ─── ISSUE 5: Inter-cluster similarity baseline ───────────────────────────────
print("\n" + "=" * 70)
print("Issue 5: Inter-cluster centroid similarity (EDGE/unconstrained/risk)")
print("=" * 70)


def get_embedding(nid):
    # node_attrs uses integer (FalkorDB) keys — convert str IDs to int
    try:
        key = int(nid)
    except (ValueError, TypeError):
        key = nid
    attrs = node_attrs.get(key, {})
    emb = attrs.get("embedding")
    if emb is None:
        return None
    if isinstance(emb, (list, np.ndarray)):
        v = np.array(emb, dtype=np.float32)
    elif isinstance(emb, str):
        cleaned = emb.strip().lstrip("<").rstrip(">")
        v = np.fromstring(cleaned, sep=",", dtype=np.float32)
    else:
        return None
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return None
    return v / norm


# Find EDGE/unconstrained/risk clusters with agglomerative algo
target_key = ("EDGE", "unconstrained", "risk", "agglomerative")
clusters_risk_edge = {}
for key, members in cluster_memberships.items():
    if len(key) == 5:
        ec, mode, nt, algo, cid = key
        if (
            str(ec).upper() == "EDGE"
            and mode == "unconstrained"
            and nt == "risk"
            and algo == "agglomerative"
        ):
            clusters_risk_edge[cid] = members

print(f"  Found {len(clusters_risk_edge)} EDGE/unconstrained/risk clusters")

# Compute cluster centroids
centroids = {}
for cid, members in clusters_risk_edge.items():
    embs = [get_embedding(m) for m in members]
    embs = [e for e in embs if e is not None]
    if embs:
        centroid = np.mean(embs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroids[cid] = centroid / norm

print(f"  Centroids computed: {len(centroids)}")

# Build 40×40 pairwise cosine similarity matrix (off-diagonal)
cids = sorted(centroids.keys())
n = len(cids)
off_diag_sims = []
for i, ci in enumerate(cids):
    for j, cj in enumerate(cids):
        if i < j:
            sim = float(np.dot(centroids[ci], centroids[cj]))
            off_diag_sims.append(sim)

off_diag_sims = np.array(off_diag_sims)
n_pairs = len(off_diag_sims)
mean_sim = float(np.mean(off_diag_sims))
min_sim = float(np.min(off_diag_sims))
max_sim = float(np.max(off_diag_sims))
pct_above_09 = float(100 * np.mean(off_diag_sims > 0.9))
pct_above_08 = float(100 * np.mean(off_diag_sims > 0.8))

print(f"\n  Off-diagonal pairs: {n_pairs:,}  (expected: {n * (n - 1) // 2})")
print(f"  Mean sim: {mean_sim:.4f}")
print(f"  Range:    [{min_sim:.4f}, {max_sim:.4f}]")
print(f"  % pairs > 0.9: {pct_above_09:.2f}%")
print(f"  % pairs > 0.8: {pct_above_08:.2f}%")

results["issue5_inter_cluster_sim"] = {
    "config": "EDGE/unconstrained/risk/agglomerative",
    "n_clusters": len(centroids),
    "n_off_diagonal_pairs": n_pairs,
    "mean_sim": round(mean_sim, 4),
    "min_sim": round(min_sim, 4),
    "max_sim": round(max_sim, 4),
    "pct_pairs_above_0.9": round(pct_above_09, 2),
    "pct_pairs_above_0.8": round(pct_above_08, 2),
}


# ─── ISSUE 4: 107× claim at consistent threshold ─────────────────────────────
print("\n" + "=" * 70)
print("Issue 4: Recompute centroid-stability ratio at consistent thresholds")
print("=" * 70)

# Load centroid similarity CSV
csim_csv = STEP2_DIR / "cluster_centroid_similarity.csv"
if csim_csv.exists():
    df_csim = pd.read_csv(csim_csv)
    print(f"  Loaded {len(df_csim):,} rows from cluster_centroid_similarity.csv")
    print(f"  Columns: {list(df_csim.columns)}")

    # high_stable_pct is % of migrating nodes with centroid_sim > threshold
    # pct_above_09 (random) from Issue 5
    # Find the column for the migration stability measure
    # The document claims "96.1% of migrations land in cluster with >0.8 centroid similarity"
    # and "only 0.9% of random cluster pairs are >0.9 similar"

    # Compute ratio at each threshold
    issue4 = {}
    for col in df_csim.columns:
        if "high_stable" in col or "pct" in col.lower():
            print(
                f"  Column '{col}': mean={df_csim[col].mean():.4f}, sample={df_csim[col].head(3).tolist()}"
            )

    # Key check: what threshold does "high_stable_pct" use?
    # From step2b code: high_stable_pct = (cs > 0.8).mean() where cs are per-node centroid sims
    # The 96.1% is fraction of migrations with centroid_sim > 0.8
    # The 0.9% is fraction of RANDOM cluster pairs with centroid_sim > 0.9 (from Issue 5)

    # For consistent comparison at 0.8:
    pct_random_above_08 = pct_above_08
    # For consistent comparison at 0.9:
    pct_random_above_09 = pct_above_09

    # "high_stable_pct" from the CSV (should be >0.8 for migrating nodes)
    if "high_stable_pct" in df_csim.columns:
        mean_high_stable = (
            df_csim["high_stable_pct"].mean() * 100
        )  # convert if fraction
        if mean_high_stable < 1.0:  # it's a fraction
            mean_high_stable *= 100
        print(
            f"\n  Mean high_stable_pct (migration centroid_sim > 0.8): {mean_high_stable:.1f}%"
        )
        print(f"  Random cluster pairs > 0.8: {pct_random_above_08:.2f}%")
        print(f"  Random cluster pairs > 0.9: {pct_random_above_09:.2f}%")

        if pct_random_above_08 > 0:
            ratio_at_08 = mean_high_stable / pct_random_above_08
            print(
                f"\n  Ratio at consistent 0.8 threshold: {ratio_at_08:.1f}× ({mean_high_stable:.1f}% vs {pct_random_above_08:.2f}%)"
            )
        if pct_random_above_09 > 0:
            ratio_at_09 = mean_high_stable / pct_random_above_09
            print(
                f"  Ratio at mixed thresholds (0.8 migration / 0.9 random): {ratio_at_09:.1f}× [MISLEADING — mixed thresholds]"
            )

        issue4 = {
            "migration_pct_centroid_sim_above_08": round(mean_high_stable, 1),
            "random_pairs_pct_above_08": round(pct_random_above_08, 2),
            "random_pairs_pct_above_09": round(pct_random_above_09, 2),
            "ratio_consistent_08": round(mean_high_stable / pct_random_above_08, 1)
            if pct_random_above_08 > 0
            else None,
            "original_107x_was_mixed_thresholds": True,
        }
    else:
        print(f"  high_stable_pct column not found; columns: {list(df_csim.columns)}")
        issue4 = {"error": "high_stable_pct column not found"}

    results["issue4_107x_recalculation"] = issue4
else:
    print(f"  WARNING: {csim_csv} not found")
    results["issue4_107x_recalculation"] = {
        "error": "centroid_similarity CSV not found"
    }


# ─── ISSUE 7: Inter-hub cosine similarities ───────────────────────────────────
print("\n" + "=" * 70)
print("Issue 7: Pairwise cosine similarities between top-5 hub nodes")
print("=" * 70)

hub_csv = STEP2_DIR / "hub_quality_metrics.csv"
if hub_csv.exists():
    df_hubs = pd.read_csv(hub_csv, dtype={"hub_node_id": str})
    # Filter to primary cut: edge_config=0.9, mode=both, node_type=risk
    df_primary = df_hubs[
        (df_hubs["edge_config"].astype(str) == "0.9") & (df_hubs["mode"] == "both")
    ]
    if "degree_sim_0.90" in df_primary.columns:
        top5 = df_primary.nlargest(5, "degree_sim_0.90")[
            ["hub_node_id", "hub_name", "degree_sim_0.90"]
        ].reset_index(drop=True)
        print("\n  Top-5 hubs by SIM≥0.9 degree (edge_config=0.9, mode=both):")
        for _, row in top5.iterrows():
            print(
                f"    [{row['hub_node_id']}] {str(row['hub_name'])[:60]:60s}  deg={row['degree_sim_0.90']}"
            )

        # Get embeddings for top-5
        hub_embs = {}
        for _, row in top5.iterrows():
            nid = str(row["hub_node_id"])
            emb = get_embedding(nid)
            if emb is not None:
                hub_embs[nid] = emb
            else:
                print(f"    WARNING: no embedding for node {nid}")

        print(f"\n  Embeddings found: {len(hub_embs)}/5")

        # Compute pairwise similarities
        pairwise = {}
        nids = list(hub_embs.keys())
        sim_values = []
        for i, ni in enumerate(nids):
            for j, nj in enumerate(nids):
                if i < j:
                    sim = float(np.dot(hub_embs[ni], hub_embs[nj]))
                    label = f"{ni}_vs_{nj}"
                    name_i = str(top5[top5["hub_node_id"] == ni]["hub_name"].values[0])[
                        :40
                    ]
                    name_j = str(top5[top5["hub_node_id"] == nj]["hub_name"].values[0])[
                        :40
                    ]
                    pairwise[label] = {
                        "sim": round(sim, 4),
                        "node_i": name_i,
                        "node_j": name_j,
                    }
                    sim_values.append(sim)
                    print(f"    {ni} vs {nj}: cos_sim = {sim:.4f}")

        if sim_values:
            print(f"\n  Range: [{min(sim_values):.4f}, {max(sim_values):.4f}]")
            results["issue7_inter_hub_cosine"] = {
                "top5_hubs": top5[
                    ["hub_node_id", "hub_name", "degree_sim_0.90"]
                ].to_dict("records"),
                "pairwise": pairwise,
                "min_sim": round(min(sim_values), 4),
                "max_sim": round(max(sim_values), 4),
                "mean_sim": round(float(np.mean(sim_values)), 4),
            }
        else:
            results["issue7_inter_hub_cosine"] = {"error": "no pairwise sims computed"}
    else:
        print(f"  degree_sim_0.90 not found; columns: {list(df_primary.columns)}")
        results["issue7_inter_hub_cosine"] = {
            "error": "degree_sim_0.90 column not found"
        }
else:
    print(f"  WARNING: {hub_csv} not found")
    results["issue7_inter_hub_cosine"] = {"error": "hub_quality_metrics.csv not found"}


# ─── ISSUE 8: Paths below ≥5 hop minimum ─────────────────────────────────────
print("\n" + "=" * 70)
print("Issue 8: Paths below ≥5-hop minimum across all path files")
print("=" * 70)

total_paths = 0
short_paths = 0  # < 5 hops (i.e., path length in nodes < 6, meaning < 5 edges)
by_file = {}

for pf in sorted(PATHS_DIR.glob("paths_*.jsonl")):
    n_total, n_short = 0, 0
    with open(pf) as f:
        for line in f:
            rec = json.loads(line)
            path = rec.get("path", [])
            if isinstance(path, str):
                path = json.loads(path)
            n_nodes = len(path)
            n_hops = n_nodes - 1  # hops = edges in path
            n_total += 1
            if n_hops < 5:
                n_short += 1
    by_file[pf.name] = {
        "total": n_total,
        "short": n_short,
        "pct_short": round(100 * n_short / n_total, 2) if n_total > 0 else 0,
    }
    total_paths += n_total
    short_paths += n_short
    if n_short > 0:
        print(
            f"  {pf.name:<55}: {n_short}/{n_total} short ({100 * n_short / n_total:.1f}%)"
        )

overall_pct = 100 * short_paths / total_paths if total_paths > 0 else 0
print(
    f"\n  TOTAL: {short_paths:,}/{total_paths:,} paths have <5 hops = {overall_pct:.2f}% removed by ≥5-hop filter"
)

# Also check EDGE-only separately
edge_only_total = sum(v["total"] for k, v in by_file.items() if "edge_only" in k)
edge_only_short = sum(v["short"] for k, v in by_file.items() if "edge_only" in k)
if edge_only_total > 0:
    print(
        f"  EDGE-only files: {edge_only_short}/{edge_only_total} short = {100 * edge_only_short / edge_only_total:.2f}%"
    )

results["issue8_hop_filter"] = {
    "total_paths_all_files": total_paths,
    "paths_below_5_hops": short_paths,
    "pct_below_5_hops": round(overall_pct, 2),
    "edge_only_total": edge_only_total,
    "edge_only_short": edge_only_short,
    "by_file": by_file,
}


# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Saving results to {OUT_FILE} …")


# Convert any numpy types for JSON serialisation
def to_json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj


with open(OUT_FILE, "w") as f:
    json.dump(to_json_safe(results), f, indent=2)
print("  Saved.")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Issue 2 — Valid-pathway SIM edge counts: {list(issue2.keys())}")
print(
    f"  Issue 5 — Inter-cluster mean sim: {results.get('issue5_inter_cluster_sim', {}).get('mean_sim')}"
)
print(
    f"  Issue 5 — % pairs > 0.9: {results.get('issue5_inter_cluster_sim', {}).get('pct_pairs_above_0.9')}%"
)
print(
    f"  Issue 7 — Inter-hub range: [{results.get('issue7_inter_hub_cosine', {}).get('min_sim')}, {results.get('issue7_inter_hub_cosine', {}).get('max_sim')}]"
)
print(f"  Issue 8 — Paths below 5 hops: {overall_pct:.2f}%")
print("Done.")
