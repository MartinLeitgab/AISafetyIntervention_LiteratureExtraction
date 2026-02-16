"""
Constrained modes - Phase 1 approach with in-memory graph
NOW INCLUDES EDGE-ONLY BASELINE
"""

import matplotlib

matplotlib.use("Agg")

import redis
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque, Counter
import json
import pickle
import os
import gc
import time

client = redis.Redis(host="localhost", port=6379, decode_responses=True)
graph = "AISafetyIntervention"


def query(q, timeout=120000):
    result = client.execute_command("GRAPH.QUERY", graph, q, "--timeout", str(timeout))
    return result[1] if len(result) > 1 else []


CAT_ORDER = [
    "risk",
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
    "intervention",
]
CAT_INDEX = {cat: i for i, cat in enumerate(CAT_ORDER)}

GLOBAL_NODE_CATEGORIES = {}
ALL_INTERVENTION_IDS = set()


def load_graph(min_conf, add_sim=False, sim_thresh=None):
    """Load graph into memory - Phase 1 approach"""
    adj, edge_map = defaultdict(set), {}

    q = "MATCH (n) RETURN min(id(n)), max(id(n))"
    result = query(q)
    min_id, max_id = int(result[0][0]), int(result[0][1])

    print("    Loading EDGE edges...", flush=True)
    cur, batch = min_id, 2000
    edge_count = 0
    while cur <= max_id:
        q = f"MATCH (n)-[e:EDGE]-(m) WHERE id(n) >= {cur} AND id(n) < {cur + batch} AND id(m) > id(n) AND e.edge_confidence >= {min_conf} RETURN id(n), id(m), e.type"
        for row in query(q):
            n1, n2 = int(row[0]), int(row[1])
            edge_type = row[2] if row[2] else "EDGE"
            adj[n1].add(n2)
            adj[n2].add(n1)
            edge_map[(n1, n2)] = edge_type
            edge_map[(n2, n1)] = edge_type
            edge_count += 1
        cur += batch
    print(f"      {edge_count:,} EDGE edges", flush=True)

    if add_sim and sim_thresh:
        print(f"    Loading SIMILARITY edges (≥{sim_thresh})...", flush=True)
        euclidean_thresh = np.sqrt(2 * (1 - sim_thresh))
        cur = min_id
        sim_count = 0
        while cur <= max_id:
            q = f"MATCH (n)-[e:SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST]-(m) WHERE id(n) >= {cur} AND id(n) < {cur + batch} AND id(m) > id(n) AND e.score < {euclidean_thresh} RETURN id(n), id(m)"
            for row in query(q):
                n1, n2 = int(row[0]), int(row[1])
                adj[n1].add(n2)
                adj[n2].add(n1)
                edge_map[(n1, n2)] = "SIMILARITY"
                edge_map[(n2, n1)] = "SIMILARITY"
                sim_count += 1
            cur += batch
        print(f"      {sim_count:,} SIMILARITY edges", flush=True)

    return adj, edge_map


def is_monotonic_step(from_cat, to_cat):
    if to_cat == "intervention":
        return True
    if from_cat != "risk" and to_cat == "risk":
        return False
    if from_cat in CAT_INDEX and to_cat in CAT_INDEX:
        return CAT_INDEX[to_cat] >= CAT_INDEX[from_cat]
    return True


def extract_mode(mode, all_sources, all_int_ids, adj, threshold_label):
    """Extract mode - Phase 1 approach: process one risk at a time"""

    output_file = f"paths_{mode}_{threshold_label}.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)

    all_pathway_nodes = set()
    total_paths = 0
    all_compositions = defaultdict(lambda: defaultdict(int))

    source_items = list(all_sources.items())
    total_sources = len(source_items)

    print(
        f"\n  Processing {total_sources} sources (Phase 1 approach: one risk at a time)...",
        flush=True,
    )

    start_time = time.time()
    last_progress = time.time()
    processed_sources = 0

    with open(output_file, "a") as f:
        for source_id, source_data in source_items:
            processed_sources += 1

            # Progress every 10 seconds
            now = time.time()
            if now - last_progress >= 10:
                elapsed = now - start_time
                rate = processed_sources / elapsed if elapsed > 0 else 0
                eta_min = (
                    ((total_sources - processed_sources) / rate / 60) if rate > 0 else 0
                )

                print(
                    f"    {processed_sources}/{total_sources} ({100 * processed_sources / total_sources:.1f}%), {total_paths:,} paths | ETA: {eta_min:.1f}min",
                    flush=True,
                )
                last_progress = now

            risks = source_data["risks"]

            # Process each risk separately (Phase 1 pattern)
            for risk_id in risks:
                visited = {risk_id: 0}
                queue = deque([risk_id])
                parent = {risk_id: None}

                while queue:
                    node = queue.popleft()

                    # Terminal intervention check
                    if node in all_int_ids and node != risk_id:
                        unvisited = [
                            nb for nb in adj.get(node, []) if nb not in visited
                        ]
                        has_non_int = any(nb not in all_int_ids for nb in unvisited)

                        if has_non_int or len(unvisited) == 0:
                            # Reconstruct path
                            path = []
                            curr = node
                            while curr is not None:
                                path.append(curr)
                                curr = parent.get(curr)
                            path = path[::-1]

                            # Build categories
                            cat_path = [
                                GLOBAL_NODE_CATEGORIES.get(n, "unknown") for n in path
                            ]

                            # Apply mode constraints during path validation
                            valid_path = True

                            if mode == "single_risk" or mode == "both":
                                # Check single-risk constraint
                                risk_count = sum(1 for cat in cat_path if cat == "risk")
                                if risk_count > 1:
                                    valid_path = False

                            if valid_path and (mode == "monotonic" or mode == "both"):
                                # Check monotonic constraint
                                cat_indices = []
                                for cat in cat_path:
                                    if cat in CAT_INDEX:
                                        cat_indices.append(CAT_INDEX[cat])
                                    elif cat == "intervention":
                                        cat_indices.append(len(CAT_ORDER))

                                # Check for reversals
                                for i in range(1, len(cat_indices)):
                                    if cat_indices[i] < cat_indices[i - 1]:
                                        valid_path = False
                                        break

                            if valid_path:
                                f.write(
                                    json.dumps(
                                        {
                                            "path": path,
                                            "length": len(path) - 1,
                                            "categories": cat_path,
                                        }
                                    )
                                    + "\n"
                                )

                                all_pathway_nodes.update(
                                    path
                                )  # Only nodes in accepted paths
                                total_paths += 1

                                cat_counts = Counter(cat_path)
                                for cat, cnt in cat_counts.items():
                                    all_compositions[len(path) - 1][cat] += cnt

                            continue  # Continue BFS for more paths

                    # Expand neighbors
                    for nb in adj.get(node, []):
                        if nb not in visited:
                            visited[nb] = visited[node] + 1
                            parent[nb] = node
                            queue.append(nb)

    print(f"\n  ✓ {total_paths:,} paths, {len(all_pathway_nodes):,} nodes", flush=True)

    # Degrees
    print("  Calculating degrees...", flush=True)
    risks = set()
    interventions = set()
    other_concepts = set()

    for nid in all_pathway_nodes:
        cat = GLOBAL_NODE_CATEGORIES.get(nid, "unknown")
        if cat == "risk":
            risks.add(nid)
        elif cat == "intervention":
            interventions.add(nid)
        elif cat in CAT_ORDER:
            other_concepts.add(nid)

    degrees_all = [len(adj[n]) for n in all_pathway_nodes]
    degrees_risks = [len(adj[n]) for n in risks]
    degrees_int = [len(adj[n]) for n in interventions]
    degrees_other = [len(adj[n]) for n in other_concepts]

    # Read lengths
    path_lengths = []
    with open(output_file, "r") as f:
        for line in f:
            path_lengths.append(json.loads(line)["length"])

    return {
        "path_file": output_file,
        "n_paths": total_paths,
        "n_nodes": len(all_pathway_nodes),
        "path_lengths": path_lengths,
        "compositions": dict(all_compositions),
        "degrees_all": degrees_all,
        "degrees_risks": degrees_risks,
        "degrees_int": degrees_int,
        "degrees_other": degrees_other,
    }


# MAIN
print("=" * 80, flush=True)
print("CONSTRAINED MODES - GRAPH IN MEMORY (WITH EDGE-ONLY)", flush=True)
print("=" * 80, flush=True)

GLOBAL_CACHE = "global_cache_constrained.pkl"
print("\nLoading cache...", flush=True)
with open(GLOBAL_CACHE, "rb") as f:
    cache = pickle.load(f)
GLOBAL_NODE_CATEGORIES = cache["categories"]
ALL_INTERVENTION_IDS = cache["interventions"]
print("✓ Loaded", flush=True)

print("\nLoading sources...", flush=True)
with open("source_pathways_final.json") as f:
    source_data = json.load(f)
all_mature = list(
    set(int(sid) for sid in source_data["mature"]["conf>=1"]["per_source"].keys())
)

q = "MATCH (s:Source) RETURN min(id(s)), max(id(s))"
result = query(q)
min_s, max_s = int(result[0][0]), int(result[0][1])

source_nodes = defaultdict(lambda: {"ints": [], "risks": []})
cur_s = min_s
batch_s = 1000

while cur_s <= max_s:
    q = f"MATCH (s:Source)<-[:FROM]-(i:Intervention) WHERE id(s) >= {cur_s} AND id(s) < {cur_s + batch_s} AND i.intervention_maturity >= 3 RETURN id(s), id(i)"
    for row in query(q):
        sid = int(row[0])
        if sid in all_mature:
            source_nodes[sid]["ints"].append(int(row[1]))

    q = f"MATCH (s:Source)<-[:FROM]-(r:Concept) WHERE id(s) >= {cur_s} AND id(s) < {cur_s + batch_s} AND r.concept_category = 'risk' RETURN id(s), id(r)"
    for row in query(q):
        sid = int(row[0])
        if sid in all_mature:
            source_nodes[sid]["risks"].append(int(row[1]))

    cur_s += batch_s

source_nodes = {sid: data for sid, data in source_nodes.items() if sid in all_mature}
print(f"  ✓ {len(source_nodes)} sources", flush=True)

print("\nLoading graph into memory...", flush=True)

# Process EDGE-only + similarity thresholds
THRESHOLDS = ["EDGE", 0.80, 0.85, 0.90, 0.95]
MODES = ["unconstrained", "single_risk", "monotonic", "both"]

all_results = {}  # {threshold: {mode: results}}

for threshold in THRESHOLDS:
    print(f"\n{'=' * 80}", flush=True)
    print(
        f"THRESHOLD: {'EDGE-only' if threshold == 'EDGE' else f'SIM≥{threshold}'}",
        flush=True,
    )
    print("=" * 80, flush=True)

    # Generate labels for filenames
    threshold_label = "edge_only" if threshold == "EDGE" else f"sim{threshold}"

    # Check if all modes for this threshold exist
    threshold_complete = all(
        os.path.exists(f"checkpoint_t{threshold_label}_mode_{mode}.pkl")
        for mode in MODES
    )

    if threshold_complete:
        print(f"✓ All modes complete for {threshold}, loading...", flush=True)
        all_results[threshold] = {}
        for mode in MODES:
            with open(f"checkpoint_t{threshold_label}_mode_{mode}.pkl", "rb") as f:
                all_results[threshold][mode] = pickle.load(f)
        continue

    # Load graph for this threshold
    if threshold == "EDGE":
        print("Loading graph (EDGE-only, conf≥3)...", flush=True)
        adj, edge_map = load_graph(3, False, None)
    else:
        print(f"Loading graph (EDGE + SIM≥{threshold})...", flush=True)
        adj, edge_map = load_graph(3, True, threshold)
    print("✓ Graph loaded", flush=True)

    all_results[threshold] = {}

    # Extract all modes for this threshold
    for mode in MODES:
        ckpt = f"checkpoint_t{threshold_label}_mode_{mode}.pkl"

        if os.path.exists(ckpt):
            print(f"\n  ✓ {mode.upper()} (checkpoint)", flush=True)
            with open(ckpt, "rb") as f:
                all_results[threshold][mode] = pickle.load(f)
        else:
            print(f"\n  {mode.upper()}", flush=True)
            print("  " + "=" * 78, flush=True)

            result = extract_mode(
                mode, source_nodes, ALL_INTERVENTION_IDS, adj, threshold_label
            )
            all_results[threshold][mode] = result

            with open(ckpt, "wb") as f:
                pickle.dump(result, f)
            print("  ✓ Checkpointed", flush=True)
            gc.collect()

    # Unload graph to free memory
    del adj, edge_map
    gc.collect()
    print(f"\n✓ Threshold {threshold} complete, graph unloaded", flush=True)

# Stats
print(f"\n{'=' * 80}", flush=True)
print("STATISTICS", flush=True)
print("=" * 80, flush=True)

for threshold in THRESHOLDS:
    print(f"\n{'=' * 80}", flush=True)
    print(
        f"THRESHOLD: {'EDGE-only' if threshold == 'EDGE' else f'SIM≥{threshold}'}",
        flush=True,
    )
    print("=" * 80, flush=True)

    for mode in MODES:
        r = all_results[threshold][mode]
        print(f"\n{mode.upper()}:", flush=True)
        print(f"  Paths: {r['n_paths']:,}", flush=True)
        print(f"  Nodes: {r['n_nodes']:,}", flush=True)
        if r["path_lengths"]:
            print(
                f"  Length: min={min(r['path_lengths'])}, max={max(r['path_lengths'])}, median={np.median(r['path_lengths']):.1f}",
                flush=True,
            )

# Plots
print(f"\n{'=' * 80}", flush=True)
print("PLOTS", flush=True)
print("=" * 80, flush=True)

colors = {
    "unconstrained": "#9B59B6",
    "single_risk": "#3498DB",
    "monotonic": "#2ECC71",
    "both": "#E74C3C",
}

# Path length distributions (2x3 grid: EDGE + 4 similarity thresholds, last empty)
print("Generating consolidated path length plot...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(24, 14))

for idx, threshold in enumerate(THRESHOLDS):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    mode_results = all_results[threshold]

    # Plot EDGE-only in gray first (reference) for similarity thresholds only
    if threshold != "EDGE":
        for mode in MODES:
            dist = Counter(all_results["EDGE"][mode]["path_lengths"])
            if dist:
                lengths, counts = zip(*sorted(dist.items()))
                ax.loglog(
                    lengths,
                    counts,
                    "-",
                    color="#999999",
                    alpha=0.4,
                    linewidth=2,
                    zorder=1,
                )  # Gray behind colored lines

    # Plot current threshold's modes (colored)
    for mode in MODES:
        dist = Counter(mode_results[mode]["path_lengths"])
        if dist:
            lengths, counts = zip(*sorted(dist.items()))
            ax.loglog(
                lengths,
                counts,
                "o-",
                color=colors[mode],
                alpha=0.7,
                label=mode.replace("_", " ").title(),
                markersize=5,
                linewidth=2,
                zorder=5,
            )

    ax.set_xlabel("Path Length (hops)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Paths", fontsize=12, fontweight="bold")
    title = "EDGE-only" if threshold == "EDGE" else f"SIM≥{threshold}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, which="both")

# Hide last subplot
axes[1, 2].axis("off")

plt.suptitle(
    "Path Length Distributions: EDGE-only + All Similarity Thresholds",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.savefig(
    "constrained_modes_path_lengths_all_with_edge.png", dpi=300, bbox_inches="tight"
)
print("  ✓ constrained_modes_path_lengths_all_with_edge.png", flush=True)

# Heatmaps - EDGE-only (2x2 grid for 4 modes)
print("\nGenerating EDGE-only heatmap...", flush=True)
bins = list(range(1, 21)) + [">20"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

mode_results_edge = all_results["EDGE"]

for idx, mode in enumerate(MODES):
    ax = axes[idx]
    mat = np.zeros((len(bins), len(CAT_ORDER)))
    comps_dict = mode_results_edge[mode]["compositions"]

    for length, cat_counts in comps_dict.items():
        bin_idx = min(length - 1, len(bins) - 1)
        for cat, cnt in cat_counts.items():
            if cat in CAT_ORDER:
                mat[bin_idx, CAT_ORDER.index(cat)] += cnt

    counts = [0] * len(bins)
    for length in mode_results_edge[mode]["path_lengths"]:
        bin_idx = min(length - 1, len(bins) - 1)
        counts[bin_idx] += 1

    for i, cnt in enumerate(counts):
        if cnt > 0:
            mat[i, :] /= cnt

    im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=5)
    ax.set_xticks(np.arange(len(CAT_ORDER)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in CAT_ORDER], fontsize=8)
    ax.set_yticks(np.arange(len(bins)))
    ax.set_yticklabels(bins, fontsize=9)
    ax.set_xlabel("Category", fontsize=10, fontweight="bold")
    ax.set_ylabel("Path Length (hops)", fontsize=10, fontweight="bold")
    ax.set_title(f"{mode.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Avg/path")

plt.suptitle("Category Heatmaps: EDGE-only", fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout()
plt.savefig("constrained_modes_heatmaps_edge_only.png", dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ constrained_modes_heatmaps_edge_only.png", flush=True)

# Heatmaps - Similarity thresholds (keep existing)
print("\nGenerating similarity heatmaps...", flush=True)

for threshold in [t for t in THRESHOLDS if t != "EDGE"]:
    print(f"  SIM≥{threshold}...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    mode_results = all_results[threshold]

    for idx, mode in enumerate(MODES):
        ax = axes[idx]
        mat = np.zeros((len(bins), len(CAT_ORDER)))
        comps_dict = mode_results[mode]["compositions"]

        for length, cat_counts in comps_dict.items():
            bin_idx = min(length - 1, len(bins) - 1)
            for cat, cnt in cat_counts.items():
                if cat in CAT_ORDER:
                    mat[bin_idx, CAT_ORDER.index(cat)] += cnt

        counts = [0] * len(bins)
        for length in mode_results[mode]["path_lengths"]:
            bin_idx = min(length - 1, len(bins) - 1)
            counts[bin_idx] += 1

        for i, cnt in enumerate(counts):
            if cnt > 0:
                mat[i, :] /= cnt

        im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=5)
        ax.set_xticks(np.arange(len(CAT_ORDER)))
        ax.set_xticklabels([c.replace(" ", "\n") for c in CAT_ORDER], fontsize=8)
        ax.set_yticks(np.arange(len(bins)))
        ax.set_yticklabels(bins, fontsize=9)
        ax.set_xlabel("Category", fontsize=10, fontweight="bold")
        ax.set_ylabel("Path Length (hops)", fontsize=10, fontweight="bold")
        ax.set_title(
            f"{mode.replace('_', ' ').title()}", fontsize=11, fontweight="bold"
        )
        plt.colorbar(im, ax=ax, label="Avg/path")

    plt.suptitle(
        f"Category Heatmaps: SIM≥{threshold}", fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(
        f"constrained_modes_heatmaps_sim{threshold}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

print("  ✓ Heatmaps complete", flush=True)

# Degree distributions (2x2 grid: all thresholds including EDGE-only)
print("\nGenerating consolidated degree plot...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# Threshold colors - EDGE is black
threshold_colors = {
    "EDGE": "#000000",
    0.80: "#9B59B6",
    0.85: "#3498DB",
    0.90: "#2ECC71",
    0.95: "#E74C3C",
}
mode_markers = {"unconstrained": "o", "single_risk": "s", "monotonic": "^", "both": "D"}
mode_lines = {"unconstrained": "-", "single_risk": "--", "monotonic": "-.", "both": ":"}

degree_types = [
    ("degrees_all", "All Pathway Nodes", axes[0, 0]),
    ("degrees_risks", "Risk Nodes", axes[0, 1]),
    ("degrees_int", "Intervention Nodes", axes[1, 0]),
    ("degrees_other", "Other Concept Nodes", axes[1, 1]),
]

for deg_key, title, ax in degree_types:
    # Plot similarity thresholds first
    for threshold in [t for t in THRESHOLDS if t != "EDGE"]:
        for mode in MODES:
            degs = all_results[threshold][mode][deg_key]
            if not degs:
                continue
            deg_counts = Counter(degs)
            d, c = zip(*sorted(deg_counts.items()))
            ax.loglog(
                d,
                c,
                marker=mode_markers[mode],
                linestyle=mode_lines[mode],
                color=threshold_colors[threshold],
                alpha=0.7,
                markersize=4,
                linewidth=1.5,
                markevery=0.1,
                zorder=5,
            )

    # Plot EDGE-only last with dashed line for all modes
    edge_line = "--"  # Force dashed for visibility
    for mode in MODES:
        degs = all_results["EDGE"][mode][deg_key]
        if not degs:
            continue
        deg_counts = Counter(degs)
        d, c = zip(*sorted(deg_counts.items()))
        ax.loglog(
            d,
            c,
            marker=mode_markers[mode],
            linestyle=edge_line,
            color="#000000",
            alpha=0.7,
            markersize=5,
            linewidth=2.0,
            markevery=1,
            zorder=10,
        )

    # Custom legend
    from matplotlib.lines import Line2D

    legend_elements = []
    for threshold in THRESHOLDS:
        label = "EDGE-only" if threshold == "EDGE" else f"SIM≥{threshold}"
        legend_elements.append(
            Line2D(
                [0], [0], color=threshold_colors[threshold], linewidth=2, label=label
            )
        )
    legend_elements.append(
        Line2D([0], [0], color="gray", linewidth=0, label="")
    )  # Spacer
    for mode in MODES:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                marker=mode_markers[mode],
                linestyle=mode_lines[mode],
                markersize=6,
                linewidth=1.5,
                label=mode.replace("_", " ").title(),
            )
        )

    ax.set_xlabel("Degree", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(handles=legend_elements, fontsize=8, loc="best", ncol=1)
    ax.grid(True, alpha=0.3, which="both")

plt.suptitle(
    "Node Degree Distributions: EDGE-only + All Similarity Thresholds",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.savefig("constrained_modes_degrees_all_with_edge.png", dpi=300, bbox_inches="tight")
print("  ✓ constrained_modes_degrees_all_with_edge.png", flush=True)

# Multi-risk statistics (including EDGE-only)
print("\nCalculating multi-risk statistics...", flush=True)

for threshold in THRESHOLDS:
    label = "EDGE-only" if threshold == "EDGE" else f"SIM≥{threshold}"
    print(f"  {label}:", flush=True)

    for mode in MODES:
        result = all_results[threshold][mode]
        multi_risk_paths = 0
        max_risks = 0

        with open(result["path_file"], "r") as f:
            for line in f:
                data = json.loads(line)
                risk_count = sum(1 for cat in data["categories"] if cat == "risk")
                if risk_count > 1:
                    multi_risk_paths += 1
                max_risks = max(max_risks, risk_count)

        total = result["n_paths"]
        pct = 100 * multi_risk_paths / total if total > 0 else 0
        print(
            f"    {mode:15s}: {multi_risk_paths:,}/{total:,} ({pct:.1f}%), max {max_risks} risks/path",
            flush=True,
        )

print("\n  ✓ All plots generated", flush=True)

# Save final results
final_cache = {
    "all_results": all_results,  # {threshold: {mode: results}}
    "thresholds": THRESHOLDS,
    "modes": MODES,
}
with open("constrained_modes_all_with_edge_final.pkl", "wb") as f:
    pickle.dump(final_cache, f)

print(f"\n{'=' * 80}", flush=True)
print("COMPLETE", flush=True)
print(
    f"Processed {len(THRESHOLDS)} thresholds × {len(MODES)} modes = {len(THRESHOLDS) * len(MODES)} extractions",
    flush=True,
)
print("=" * 80, flush=True)
