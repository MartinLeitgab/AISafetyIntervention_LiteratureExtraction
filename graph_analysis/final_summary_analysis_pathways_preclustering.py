"""
Complete pathway analysis - optimized with global category caching
Fixed: Uses ID range queries and caches all categories upfront
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

client = redis.Redis(host="localhost", port=6379, decode_responses=True)
graph = "AISafetyIntervention"


def query(q, timeout=120000):
    result = client.execute_command("GRAPH.QUERY", graph, q, "--timeout", str(timeout))
    return result[1] if len(result) > 1 else []


CACHE_FILE = "pathway_cache_final.pkl"
CAT_ORDER = [
    "risk",
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
]

# GLOBAL CATEGORY CACHE - loaded once at startup
GLOBAL_NODE_CATEGORIES = {}


def load_all_categories():
    """Load all node categories once using ID range queries (Phase 1 pattern)"""
    print("Loading all node categories globally (one-time operation)...")

    # Get node ID range
    q = "MATCH (n) RETURN min(id(n)), max(id(n))"
    result = query(q)
    min_id, max_id = int(result[0][0]), int(result[0][1])

    categories = {}
    current_id = min_id
    batch_size = 5000  # Larger batch for simple queries

    total_batches = (max_id - min_id) // batch_size + 1
    batch_num = 0

    while current_id <= max_id:
        batch_num += 1
        if batch_num % 10 == 0:
            print(
                f"  Progress: {batch_num}/{total_batches} batches ({100 * batch_num / total_batches:.1f}%)"
            )

        # Load Concept categories
        q = f"""
        MATCH (c:Concept) 
        WHERE id(c) >= {current_id} AND id(c) < {current_id + batch_size}
        RETURN id(c), c.concept_category
        """
        for row in query(q):
            if row[1]:  # Has category
                categories[int(row[0])] = row[1]

        # Load Interventions (mark as 'intervention')
        q = f"""
        MATCH (i:Intervention) 
        WHERE id(i) >= {current_id} AND id(i) < {current_id + batch_size}
        RETURN id(i)
        """
        for row in query(q):
            categories[int(row[0])] = "intervention"

        current_id += batch_size

    print(f"  ✓ Loaded {len(categories):,} node categories")

    # Validation: Check against expected totals
    q = "MATCH (c:Concept) RETURN count(c)"
    concept_count = int(query(q)[0][0])
    q = "MATCH (i:Intervention) RETURN count(i)"
    int_count = int(query(q)[0][0])
    expected = concept_count + int_count

    if (
        len(categories) < expected * 0.95
    ):  # Allow 5% margin for nodes without categories
        print(
            f"  ⚠ WARNING: Only loaded {len(categories):,} categories but expected ~{expected:,}"
        )
    else:
        print(
            f"  ✓ Validation: {len(categories):,} categories loaded from {expected:,} total nodes"
        )

    return categories


def load_all_interventions_and_risks():
    """Load all intervention and risk IDs once using ID range queries"""
    print("Loading all intervention and risk IDs...")

    q = "MATCH (n) RETURN min(id(n)), max(id(n))"
    result = query(q)
    min_id, max_id = int(result[0][0]), int(result[0][1])

    intervention_ids = set()
    risk_ids = set()
    current_id = min_id
    batch_size = 5000

    while current_id <= max_id:
        # Load interventions with maturity >= 3
        q = f"""
        MATCH (i:Intervention)
        WHERE id(i) >= {current_id} AND id(i) < {current_id + batch_size}
          AND i.intervention_maturity >= 3
        RETURN id(i)
        """
        for row in query(q):
            intervention_ids.add(int(row[0]))

        # Load risks
        q = f"""
        MATCH (c:Concept)
        WHERE id(c) >= {current_id} AND id(c) < {current_id + batch_size}
          AND c.concept_category = 'risk'
        RETURN id(c)
        """
        for row in query(q):
            risk_ids.add(int(row[0]))

        current_id += batch_size

    print(f"  ✓ Loaded {len(intervention_ids):,} interventions (maturity≥3)")
    print(f"  ✓ Loaded {len(risk_ids):,} risks")

    # Validation
    q = "MATCH (i:Intervention) WHERE i.intervention_maturity >= 3 RETURN count(i)"
    expected_int = int(query(q)[0][0])
    q = "MATCH (c:Concept) WHERE c.concept_category = 'risk' RETURN count(c)"
    expected_risk = int(query(q)[0][0])

    assert len(intervention_ids) == expected_int, (
        f"Intervention count mismatch: {len(intervention_ids)} vs {expected_int}"
    )
    assert len(risk_ids) == expected_risk, (
        f"Risk count mismatch: {len(risk_ids)} vs {expected_risk}"
    )
    print("  ✓ Validation passed: All interventions and risks loaded")

    return intervention_ids, risk_ids


def load_graph(min_conf, add_sim=False, sim_thresh=None):
    """Load graph adjacency using ID range queries (Phase 1 pattern)"""
    adj, edge_map = defaultdict(set), {}

    q = "MATCH (n) RETURN min(id(n)), max(id(n))"
    result = query(q)
    min_id, max_id = int(result[0][0]), int(result[0][1])

    # EDGE edges
    print("    Loading EDGE edges...")
    cur, batch = min_id, 2000
    edge_count = 0
    while cur <= max_id:
        q = f"""
        MATCH (n)-[e:EDGE]-(m) 
        WHERE id(n) >= {cur} AND id(n) < {cur + batch} 
          AND id(m) > id(n) 
          AND e.edge_confidence >= {min_conf} 
        RETURN id(n), id(m), e.type
        """
        for row in query(q):
            n1, n2 = int(row[0]), int(row[1])
            edge_type = row[2] if row[2] else "EDGE"
            adj[n1].add(n2)
            adj[n2].add(n1)
            edge_map[(n1, n2)] = edge_type
            edge_map[(n2, n1)] = edge_type
            edge_count += 1
        cur += batch
    print(f"      Loaded {edge_count:,} EDGE edges")

    # SIMILARITY edges
    if add_sim and sim_thresh:
        print(f"    Loading SIMILARITY edges (threshold {sim_thresh})...")
        euclidean_thresh = np.sqrt(2 * (1 - sim_thresh))
        cur = min_id
        sim_count = 0
        while cur <= max_id:
            q = f"""
            MATCH (n)-[e:SIMILARITY_ABOVE_POINT_EIGHT_2150_NEAREST]-(m) 
            WHERE id(n) >= {cur} AND id(n) < {cur + batch} 
              AND id(m) > id(n) 
              AND e.score < {euclidean_thresh} 
            RETURN id(n), id(m)
            """
            for row in query(q):
                n1, n2 = int(row[0]), int(row[1])
                adj[n1].add(n2)
                adj[n2].add(n1)
                edge_map[(n1, n2)] = "SIMILARITY"
                edge_map[(n2, n1)] = "SIMILARITY"
                sim_count += 1
            cur += batch
        print(f"      Loaded {sim_count:,} SIMILARITY edges")

    return adj, edge_map


def extract_pathways(
    min_conf, use_all_mat, adj, edge_map, source_filter=None, all_int_ids=None
):
    """Extract pathways using global category cache"""
    mat_key = "all" if use_all_mat else "mature"

    with open("source_pathways_final.json") as f:
        source_data = json.load(f)

    if source_filter:
        source_ids = source_filter
    else:
        per_source = source_data[mat_key][f"conf>={min_conf}"]["per_source"]
        source_ids = [int(sid) for sid, count in per_source.items()]

    # Preload source->intervention/risk mappings using ID range queries
    print("    Preloading source mappings...")
    source_nodes = defaultdict(lambda: {"ints": [], "risks": []})

    # Get source ID range
    q = "MATCH (s:Source) RETURN min(id(s)), max(id(s))"
    result = query(q)
    min_s, max_s = int(result[0][0]), int(result[0][1])

    cur_s = min_s
    batch_s = 1000
    while cur_s <= max_s:
        # Get interventions from sources in this range
        q = f"""
        MATCH (s:Source)<-[:FROM]-(i:Intervention) 
        WHERE id(s) >= {cur_s} AND id(s) < {cur_s + batch_s}
        """
        if not use_all_mat:
            q += " AND i.intervention_maturity >= 3"
        q += " RETURN id(s), id(i)"

        for row in query(q):
            sid = int(row[0])
            if sid in source_ids or not source_filter:
                source_nodes[sid]["ints"].append(int(row[1]))

        # Get risks from sources in this range
        q = f"""
        MATCH (s:Source)<-[:FROM]-(r:Concept) 
        WHERE id(s) >= {cur_s} AND id(s) < {cur_s + batch_s}
          AND r.concept_category = 'risk' 
        RETURN id(s), id(r)
        """
        for row in query(q):
            sid = int(row[0])
            if sid in source_ids or not source_filter:
                source_nodes[sid]["risks"].append(int(row[1]))

        cur_s += batch_s

    # Filter to requested sources
    if source_filter:
        source_nodes = {
            sid: data for sid, data in source_nodes.items() if sid in source_filter
        }

    print(f"      Loaded {len(source_nodes)} sources")

    # Validation: check we got all expected sources
    if source_filter:
        if len(source_nodes) != len(source_filter):
            print(
                f"  ⚠ WARNING: Loaded {len(source_nodes)} sources but expected {len(source_filter)}"
            )

    # Count total interventions and risks
    total_ints = sum(len(data["ints"]) for data in source_nodes.values())
    total_risks = sum(len(data["risks"]) for data in source_nodes.values())
    print(f"      {total_ints:,} interventions, {total_risks:,} risks across sources")

    # Build intervention ID set for fast lookup
    if all_int_ids is None:
        all_int_ids = set()
        for data in source_nodes.values():
            all_int_ids.update(data["ints"])

    # BFS - START AT RISKS
    pathway_nodes, path_lengths, long_paths, comps = set(), [], [], []
    all_paths = []

    import time

    start_time = time.time()
    last_progress = time.time()
    total_sources = len(source_nodes)
    processed_sources = 0

    for source_id, data in source_nodes.items():
        processed_sources += 1

        # Progress reporting every 10 seconds
        now = time.time()
        if now - last_progress >= 10:
            elapsed = now - start_time
            rate = processed_sources / elapsed if elapsed > 0 else 0
            remaining = total_sources - processed_sources
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_min = eta_seconds / 60

            print(
                f"      BFS Progress: {processed_sources}/{total_sources} sources ({100 * processed_sources / total_sources:.1f}%), {len(all_paths):,} paths, {len(pathway_nodes):,} nodes | ETA: {eta_min:.1f}min"
            )
            last_progress = now
        ints, risks = set(data["ints"]), data["risks"]

        # Start from RISKS
        for risk_id in risks:
            visited, queue, parent = {risk_id: 0}, deque([risk_id]), {risk_id: None}
            pathway_nodes.add(risk_id)

            while queue:
                node = queue.popleft()

                # Check if we're at an intervention with non-intervention neighbors (terminal point)
                if node in all_int_ids and node != risk_id:
                    unvisited_neighbors = [
                        nb for nb in adj.get(node, []) if nb not in visited
                    ]
                    has_non_int_neighbor = any(
                        nb not in all_int_ids for nb in unvisited_neighbors
                    )

                    if has_non_int_neighbor or len(unvisited_neighbors) == 0:
                        # Terminal intervention: record path and STOP
                        path, edges = [], []
                        curr = node
                        while curr is not None:
                            path.append(curr)
                            if parent.get(curr) is not None:
                                p = parent[curr]
                                edges.append(
                                    edge_map.get(
                                        (min(p, curr), max(p, curr)), "unknown"
                                    )
                                )
                            curr = parent.get(curr)

                        path = path[::-1]
                        edges = edges[::-1]

                        path_lengths.append(len(path) - 1)
                        all_paths.append(
                            {
                                "path": path,
                                "edges": edges,
                                "length": len(path) - 1,
                                "source_id": source_id,
                            }
                        )
                        continue

                # Expand neighbors
                for nb in adj.get(node, []):
                    if nb not in visited:
                        visited[nb] = visited[node] + 1
                        parent[nb] = node
                        pathway_nodes.add(nb)
                        queue.append(nb)

    print(
        f"      Extracted {len(all_paths):,} paths, {len(pathway_nodes):,} unique nodes visited"
    )

    # Extract nodes that are actually in paths (not just visited during BFS)
    path_nodes_set = set()
    for p in all_paths:
        path_nodes_set.update(p["path"])

    print(
        f"      Nodes in actual paths: {len(path_nodes_set):,} ({100 * len(path_nodes_set) / len(pathway_nodes) if pathway_nodes else 0:.1f}% of visited)"
    )

    # Use GLOBAL category cache instead of querying
    print("    Building path compositions from global cache...")
    for p in all_paths:
        path_cats = [GLOBAL_NODE_CATEGORIES.get(n, "unknown") for n in p["path"]]
        cat_counts = Counter(path_cats)

        comps.append(
            {
                "length": p["length"],
                "categories": dict(cat_counts),
                "n_unique_cats": len([c for c in cat_counts if c != "intervention"]),
            }
        )

        if p["length"] > 10:
            p["path_cats"] = path_cats
            p["categories"] = dict(cat_counts)
            long_paths.append(p)

    return path_nodes_set, path_lengths, long_paths, comps


def get_attrs(nodes):
    """Get attributes using ID range queries and global cache"""
    attrs = {"lifecycle": [], "category": []}

    # Categories from global cache
    for nid in nodes:
        cat = GLOBAL_NODE_CATEGORIES.get(nid)
        if cat:
            attrs["category"].append(cat)

    # Lifecycle requires query (intervention-specific attribute)
    node_list = list(nodes)
    q = "MATCH (n) RETURN min(id(n)), max(id(n))"
    result = query(q)
    min_id, max_id = int(result[0][0]), int(result[0][1])

    cur = min_id
    batch = 5000
    while cur <= max_id:
        q = f"""
        MATCH (i:Intervention) 
        WHERE id(i) >= {cur} AND id(i) < {cur + batch}
        RETURN id(i), i.intervention_lifecycle
        """
        for row in query(q):
            nid = int(row[0])
            if nid in nodes and row[1]:
                attrs["lifecycle"].append(row[1])
        cur += batch

    return attrs


if os.path.exists(CACHE_FILE):
    print("Loading cached data...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    print("✓ Loaded")
else:
    # Check for checkpoints to resume from
    checkpoints = {
        "edge": "checkpoint_edge.pkl",
        0.80: "checkpoint_sim_0.8.pkl",
        0.85: "checkpoint_sim_0.85.pkl",
        0.90: "checkpoint_sim_0.9.pkl",
        0.95: "checkpoint_sim_0.95.pkl",
    }

    resume_data = {}
    for key, path in checkpoints.items():
        if os.path.exists(path):
            print(f"Found checkpoint: {path}")
            with open(path, "rb") as f:
                resume_data[key] = pickle.load(f)

    if resume_data:
        print(f"✓ Loaded {len(resume_data)} checkpoints, will resume incomplete work")

    print("=" * 80)
    print("INITIALIZING GLOBAL CACHES")
    print("=" * 80)

    # Load categories ONCE
    GLOBAL_NODE_CATEGORIES = load_all_categories()

    # Load intervention/risk IDs ONCE
    ALL_INTERVENTION_IDS, ALL_RISK_IDS = load_all_interventions_and_risks()

    print("\n" + "=" * 80)
    print("EXTRACTING PATHWAYS")
    print("=" * 80)

    # Check for EDGE checkpoint
    if "edge" in resume_data:
        print("\n✓ Resuming from EDGE checkpoint")
        nodes_g = resume_data["edge"]["nodes_g"]
        lens_g = resume_data["edge"]["lens_g"]
        comp_g = resume_data["edge"]["comp_g"]
        nodes_c = resume_data["edge"]["nodes_c"]
        lens_c = resume_data["edge"]["lens_c"]
        comp_c = resume_data["edge"]["comp_c"]
        long_paths = resume_data["edge"]["long_paths"]
    else:
        # Extract
        print("\nGray (all interventions)...")
        adj_g, em_g = load_graph(1, False, None)
        nodes_g, lens_g, _, comp_g = extract_pathways(
            1, True, adj_g, em_g, all_int_ids=ALL_INTERVENTION_IDS
        )

        print("\nColored (mature interventions)...")
        adj_c, em_c = load_graph(3, False, None)
        nodes_c, lens_c, long_paths, comp_c = extract_pathways(
            3, False, adj_c, em_c, all_int_ids=ALL_INTERVENTION_IDS
        )

        # CHECKPOINT: Save gray + color EDGE-only data
        print("  Checkpointing EDGE-only results...")
        with open("checkpoint_edge.pkl", "wb") as f:
            pickle.dump(
                {
                    "nodes_g": nodes_g,
                    "lens_g": lens_g,
                    "comp_g": comp_g,
                    "nodes_c": nodes_c,
                    "lens_c": lens_c,
                    "comp_c": comp_c,
                    "long_paths": long_paths,
                },
                f,
            )
        gc.collect()  # Free memory after checkpoint

    print("\nSIMILARITY thresholds...")
    # Get mature source IDs
    with open("source_pathways_final.json") as f:
        source_data = json.load(f)
    all_mature = list(
        set(int(sid) for sid in source_data["mature"]["conf>=1"]["per_source"].keys())
    )

    sim_data = {}
    sim_comps = {}
    sim_nodes_c = {}  # Nodes for color scenario only

    for t in [0.80, 0.85, 0.90, 0.95]:
        if t in resume_data:
            print(f"  ✓ Skipping threshold {t} (checkpoint exists)")
            sim_data[t] = resume_data[t]["lens"]
            sim_comps[t] = resume_data[t]["comps"]
            sim_nodes_c[t] = resume_data[t]["nodes"]
        else:
            print(f"  Threshold {t}...")
            adj_s, em_s = load_graph(3, True, t)

            # Color (mature): conf>=3, maturity>=3
            nodes_s_c, lens_s, _, comp_s = extract_pathways(
                3,
                False,
                adj_s,
                em_s,
                source_filter=all_mature,
                all_int_ids=ALL_INTERVENTION_IDS,
            )
            sim_data[t] = lens_s
            sim_comps[t] = comp_s
            sim_nodes_c[t] = nodes_s_c

            print(f"    {len(lens_s):,} paths, {len(nodes_s_c):,} nodes")

            # CHECKPOINT: Save after each threshold
            print(f"    Checkpointing threshold {t}...")
            with open(f"checkpoint_sim_{t}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "threshold": t,
                        "nodes": nodes_s_c,
                        "lens": lens_s,
                        "comps": comp_s,
                    },
                    f,
                )
            gc.collect()  # Free memory after checkpoint
            print("    Memory freed")

    print("\nGathering attributes...")
    attrs_g = get_attrs(nodes_g)
    attrs_c = get_attrs(nodes_c)

    print("Calculating degrees for all scenarios...")
    # Load graphs for degree calculation (needed when resuming from checkpoints)
    print("  Loading graphs...")
    adj_g, _ = load_graph(1, False, None)  # Gray
    adj_c, _ = load_graph(3, False, None)  # Color

    # EDGE-only degrees
    degrees_g = [len(adj_g[n]) for n in nodes_g]
    degrees_c = [len(adj_c[n]) for n in nodes_c]

    # Separate nodes by type for color scenario
    print("  Separating nodes by type...")
    risks_c = set()
    interventions_c = set()
    other_concepts_c = set()

    for nid in nodes_c:
        cat = GLOBAL_NODE_CATEGORIES.get(nid, "unknown")
        if cat == "risk":
            risks_c.add(nid)
        elif cat == "intervention":
            interventions_c.add(nid)
        elif cat in CAT_ORDER:
            other_concepts_c.add(nid)

    print(f"    Risks: {len(risks_c):,}")
    print(f"    Interventions: {len(interventions_c):,}")
    print(f"    Other concepts: {len(other_concepts_c):,}")

    # Degrees by type for EDGE-only
    degrees_risks_c = [len(adj_c[n]) for n in risks_c]
    degrees_int_c = [len(adj_c[n]) for n in interventions_c]
    degrees_other_c = [len(adj_c[n]) for n in other_concepts_c]

    print(
        f"  Gray degrees: min={min(degrees_g) if degrees_g else 'N/A'}, max={max(degrees_g) if degrees_g else 'N/A'}, zero-degree nodes={sum(1 for d in degrees_g if d == 0)}"
    )
    print(
        f"  Color degrees: min={min(degrees_c) if degrees_c else 'N/A'}, max={max(degrees_c) if degrees_c else 'N/A'}, zero-degree nodes={sum(1 for d in degrees_c if d == 0)}"
    )

    # Similarity degrees (color scenario only) - INCLUDES EDGE EDGES
    sim_degrees_c = {}
    sim_degrees_risks_c = {}
    sim_degrees_int_c = {}
    sim_degrees_other_c = {}

    # Load EDGE graph once for combining with similarity
    adj_edge_c, _ = load_graph(3, False, None)

    for t in [0.80, 0.85, 0.90, 0.95]:
        print(f"  Degrees for threshold {t} (EDGE+SIM combined)...")
        adj_sim_c, _ = load_graph(3, True, t)

        # Combine EDGE and similarity adjacencies
        adj_combined = defaultdict(set)
        for n in adj_edge_c:
            adj_combined[n].update(adj_edge_c[n])
        for n in adj_sim_c:
            adj_combined[n].update(adj_sim_c[n])

        sim_degrees_c[t] = [len(adj_combined[n]) for n in sim_nodes_c[t]]

        # Separate by type for this threshold
        risks_t = set()
        int_t = set()
        other_t = set()
        for nid in sim_nodes_c[t]:
            cat = GLOBAL_NODE_CATEGORIES.get(nid, "unknown")
            if cat == "risk":
                risks_t.add(nid)
            elif cat == "intervention":
                int_t.add(nid)
            elif cat in CAT_ORDER:
                other_t.add(nid)

        sim_degrees_risks_c[t] = [len(adj_combined[n]) for n in risks_t]
        sim_degrees_int_c[t] = [len(adj_combined[n]) for n in int_t]
        sim_degrees_other_c[t] = [len(adj_combined[n]) for n in other_t]

        # Force memory cleanup
        del adj_combined, adj_sim_c
        gc.collect()

        print(
            f"    SIM {t}: min={min(sim_degrees_c[t]) if sim_degrees_c[t] else 'N/A'}, max={max(sim_degrees_c[t]) if sim_degrees_c[t] else 'N/A'}, zero-degree nodes={sum(1 for d in sim_degrees_c[t] if d == 0)}"
        )
        print(
            f"      Risks: {len(risks_t):,}, Int: {len(int_t):,}, Other: {len(other_t):,}"
        )

    # Community detection for each node type at all thresholds (EDGE+SIM combined)
    print("\nDetecting communities (EDGE+SIM combined graphs)...")
    import networkx as nx

    def detect_communities(node_set, adj):
        """Detect communities using connected components"""
        G = nx.Graph()
        for n in node_set:
            G.add_node(n)
            for neighbor in adj.get(n, []):
                if neighbor in node_set:
                    G.add_edge(n, neighbor)
        components = list(nx.connected_components(G))
        return components

    # Load EDGE graph once for combining
    adj_edge_comm, _ = load_graph(3, False, None)

    community_info = {}
    for t in [0.80, 0.85, 0.90, 0.95]:
        print(f"  Communities at threshold {t} (EDGE+SIM)...")
        adj_sim_comm, _ = load_graph(3, True, t)

        # Combine EDGE and similarity adjacencies
        adj_combined_comm = defaultdict(set)
        for n in adj_edge_comm:
            adj_combined_comm[n].update(adj_edge_comm[n])
        for n in adj_sim_comm:
            adj_combined_comm[n].update(adj_sim_comm[n])

        # Separate nodes for this threshold
        risks_t = set()
        int_t = set()
        other_t = set()
        for nid in sim_nodes_c[t]:
            cat = GLOBAL_NODE_CATEGORIES.get(nid, "unknown")
            if cat == "risk":
                risks_t.add(nid)
            elif cat == "intervention":
                int_t.add(nid)
            elif cat in CAT_ORDER:
                other_t.add(nid)

        communities_risks = detect_communities(risks_t, adj_combined_comm)
        communities_int = detect_communities(int_t, adj_combined_comm)
        communities_other = detect_communities(other_t, adj_combined_comm)

        # Force memory cleanup
        del adj_combined_comm, adj_sim_comm
        gc.collect()

        community_info[t] = {
            "risks": {
                "n_communities": len(communities_risks),
                "sizes": sorted([len(c) for c in communities_risks], reverse=True)[:20],
            },
            "interventions": {
                "n_communities": len(communities_int),
                "sizes": sorted([len(c) for c in communities_int], reverse=True)[:20],
            },
            "other": {
                "n_communities": len(communities_other),
                "sizes": sorted([len(c) for c in communities_other], reverse=True)[:20],
            },
        }
        print(
            f"    Risks: {len(communities_risks):,}, Int: {len(communities_int):,}, Other: {len(communities_other):,}"
        )

    # PATHWAY CLUSTERING - SKIPPED (deferred to Phase 2)
    print("\nPathway mechanism clustering: SKIPPED")
    print("  Reason: BFS sampling hits 27GB limit")
    print("  Solution: Phase 2 signature-based clustering")

    pathway_mechanism_clusters = {"status": "deferred", "estimated_reduction": "8-10x"}

    cache = {
        "attrs_g": attrs_g,
        "attrs_c": attrs_c,
        "lens_g": lens_g,
        "lens_c": lens_c,
        "degrees_g": degrees_g,
        "degrees_c": degrees_c,
        "degrees_risks_c": degrees_risks_c,
        "degrees_int_c": degrees_int_c,
        "degrees_other_c": degrees_other_c,
        "sim_degrees_c": sim_degrees_c,
        "sim_degrees_risks_c": sim_degrees_risks_c,
        "sim_degrees_int_c": sim_degrees_int_c,
        "sim_degrees_other_c": sim_degrees_other_c,
        "community_info": community_info,
        "pathway_mechanism_clusters": pathway_mechanism_clusters,
        "sim_data": sim_data,
        "sim_comps": sim_comps,
        "long_paths": long_paths,
        "comp_g": comp_g,
        "comp_c": comp_c,
        "nodes_g": len(nodes_g),
        "nodes_c": len(nodes_c),
    }

    print("\nCaching results...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print("✓ Cached")

# Unpack
attrs_g, attrs_c = cache["attrs_g"], cache["attrs_c"]
lens_g, lens_c = cache["lens_g"], cache["lens_c"]
degrees_g, degrees_c = cache["degrees_g"], cache["degrees_c"]
sim_data, long_paths = cache["sim_data"], cache["long_paths"]
comp_g, comp_c = cache["comp_g"], cache["comp_c"]

# DEBUG: Analyze degree distributions from cache
print(f"\n{'=' * 80}\nDEBUG: DEGREE ANALYSIS FROM CACHE\n{'=' * 80}")
print("Gray (all interventions, EDGE-only):")
print(f"  Total nodes: {len(degrees_g):,}")
print(f"  Min degree: {min(degrees_g) if degrees_g else 'N/A'}")
print(f"  Max degree: {max(degrees_g) if degrees_g else 'N/A'}")
print(f"  Zero-degree nodes: {sum(1 for d in degrees_g if d == 0):,}")
if sum(1 for d in degrees_g if d == 0) > 0:
    print(
        f"  WARNING: {sum(1 for d in degrees_g if d == 0)} nodes with degree=0 (shouldn't exist in pathways)"
    )

print("\nColor (mature interventions, EDGE-only):")
print(f"  Total nodes: {len(degrees_c):,}")
print(f"  Min degree: {min(degrees_c) if degrees_c else 'N/A'}")
print(f"  Max degree: {max(degrees_c) if degrees_c else 'N/A'}")
print(f"  Zero-degree nodes: {sum(1 for d in degrees_c if d == 0):,}")
if sum(1 for d in degrees_c if d == 0) > 0:
    print(
        f"  WARNING: {sum(1 for d in degrees_c if d == 0)} nodes with degree=0 (shouldn't exist in pathways)"
    )

# Check similarity degrees if in cache
if "sim_degrees_c" in cache:
    print("\nSimilarity thresholds (mature):")
    for t in sorted(cache["sim_degrees_c"].keys()):
        degs = cache["sim_degrees_c"][t]
        print(
            f"  SIM≥{t}: nodes={len(degs):,}, min={min(degs) if degs else 'N/A'}, max={max(degs) if degs else 'N/A'}, zero={sum(1 for d in degs if d == 0):,}"
        )
else:
    print("\n  ⚠ No similarity degree data in cache (needs regeneration)")

print(f"{'=' * 80}\n")

# STATS
print(f"\n{'=' * 80}\nSTATS\n{'=' * 80}")
print("\nPath counts:")
print(f"  Gray (all): {len(lens_g):,} paths")
print(f"  Colored (mature): {len(lens_c):,} paths")
for t in sorted(sim_data.keys()):
    print(f"  +SIM ≥{t}: {len(sim_data[t]):,} paths")

print("\nNode counts (only nodes in actual paths):")
print(f"  Gray: {cache['nodes_g']:,} nodes")
print(f"  Colored: {cache['nodes_c']:,} nodes")

print(f"\nLifecycle distribution: {Counter(attrs_c['lifecycle'])}")
print(f"Category distribution: {Counter(attrs_c['category'])}")

if lens_c:
    print("\nPath length statistics:")
    print(f"  Min: {min(lens_c)} hops")
    print(f"  Max: {max(lens_c)} hops")
    print(f"  Median: {np.median(lens_c):.1f} hops")
    print(f"  Mean: {np.mean(lens_c):.1f} hops")

if "community_info" in cache:
    print("\nCommunity structure (mature interventions, connected components):")
    ci = cache["community_info"]
    for t in sorted(ci.keys()):
        print(f"  SIM≥{t}:")
        print(
            f"    Risks: {ci[t]['risks']['n_communities']:,} communities, top 5: {ci[t]['risks']['sizes'][:5]}"
        )
        print(
            f"    Interventions: {ci[t]['interventions']['n_communities']:,} communities, top 5: {ci[t]['interventions']['sizes'][:5]}"
        )
        print(
            f"    Other concepts: {ci[t]['other']['n_communities']:,} communities, top 5: {ci[t]['other']['sizes'][:5]}"
        )

# LONG PATHS
long_list = [p for p in long_paths if p["length"] > 10]
print(f"\n{'=' * 80}\nLONG PATHS ({len(long_list)})\n{'=' * 80}")

for i, p in enumerate(long_list):
    details = []
    for nid in p["path"]:
        q = f"MATCH (i:Intervention) WHERE id(i)={nid} RETURN i.name, i.description"
        res = query(q)
        if res and res[0][0]:
            details.append(("intervention", res[0][0], res[0][1] or ""))
        else:
            q = f"MATCH (c:Concept) WHERE id(c)={nid} RETURN c.name, c.description, c.concept_category"
            res = query(q)
            details.append(
                (res[0][2], res[0][0], res[0][1] or "")
                if res and res[0][0]
                else ("unknown", f"Node{nid}", "")
            )

    q = f"MATCH (s:Source) WHERE id(s)={p['source_id']} RETURN s.url"
    url = query(q)
    url = url[0][0] if url and url[0][0] else "Unknown"

    # Direction check
    cat_idx = []
    for cat, _, _ in details:
        if cat in CAT_ORDER:
            cat_idx.append(CAT_ORDER.index(cat))
        elif cat == "intervention":
            cat_idx.append(6)

    reversals = sum(1 for j in range(1, len(cat_idx)) if cat_idx[j] < cat_idx[j - 1])
    consistency = (
        1 - (reversals / max(len(cat_idx) - 1, 1)) if len(cat_idx) > 1 else 1.0
    )

    print(f"\n{'=' * 80}")
    print(f"Path {i + 1}: {p['length']} hops | {url}")
    print(
        f"Categories: {p['categories']} | Consistency: {consistency:.0%} ({reversals} reversals)"
    )
    print("\nRISK → INTERVENTION:")

    for j, (cat, name, desc) in enumerate(details):
        desc_short = desc[:150] + "..." if len(desc) > 150 else desc
        print(f"  {j + 1}. [{cat}] {name}")
        if desc_short:
            print(f"     {desc_short}")
        if j < len(p["edges"]):
            print(f"     --[{p['edges'][j]}]-->")

# PLOTS
print(f"\n{'=' * 80}\nGENERATING PLOTS\n{'=' * 80}")

fig = plt.figure(figsize=(18, 8))

# Lifecycle mapping
lifecycle_map = {
    1: "Model\nDesign",
    2: "Pre-\nTraining",
    3: "Fine-Tuning/\nRL",
    4: "Pre-Deploy\nTesting",
    5: "Deployment",
    6: "Other",
}

# Lifecycle - Abs
ax = plt.subplot(2, 3, 1)
life_g, life_c = Counter(attrs_g["lifecycle"]), Counter(attrs_c["lifecycle"])
cats = sorted(set(life_g.keys()) | set(life_c.keys()))
labels = [lifecycle_map.get(c, str(c)) for c in cats]
x = np.arange(len(cats))
ax.bar(x - 0.2, [life_g[c] for c in cats], 0.4, color="gray", alpha=0.6, label="All")
ax.bar(x + 0.2, [life_c[c] for c in cats], 0.4, color="steelblue", label="Mature")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Lifecycle (Abs)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Lifecycle - Rel
ax = plt.subplot(2, 3, 4)
tg, tc = sum(life_g.values()), sum(life_c.values())
if tg > 0 and tc > 0:
    ax.bar(
        x - 0.2,
        [100 * life_g[c] / tg for c in cats],
        0.4,
        color="gray",
        alpha=0.6,
        label="All",
    )
    ax.bar(
        x + 0.2,
        [100 * life_c[c] / tc for c in cats],
        0.4,
        color="steelblue",
        label="Mature",
    )
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("%", fontsize=10)
ax.set_title("Lifecycle (Rel)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Category - Abs
ax = plt.subplot(2, 3, 2)
cat_g, cat_c = Counter(attrs_g["category"]), Counter(attrs_c["category"])
cats = [c for c in CAT_ORDER if c in cat_g or c in cat_c]
x = np.arange(len(cats))
ax.bar(x - 0.2, [cat_g[c] for c in cats], 0.4, color="gray", alpha=0.6, label="All")
ax.bar(x + 0.2, [cat_c[c] for c in cats], 0.4, color="coral", label="Mature")
ax.set_xticks(x)
ax.set_xticklabels([c.replace(" ", "\n") for c in cats], fontsize=8)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Category (Abs)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Category - Rel
ax = plt.subplot(2, 3, 5)
tg, tc = sum(cat_g.values()), sum(cat_c.values())
if tg > 0 and tc > 0:
    ax.bar(
        x - 0.2,
        [100 * cat_g[c] / tg for c in cats],
        0.4,
        color="gray",
        alpha=0.6,
        label="All",
    )
    ax.bar(
        x + 0.2, [100 * cat_c[c] / tc for c in cats], 0.4, color="coral", label="Mature"
    )
ax.set_xticks(x)
ax.set_xticklabels([c.replace(" ", "\n") for c in cats], fontsize=8)
ax.set_ylabel("%", fontsize=10)
ax.set_title("Category (Rel)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "pathway_conceptcategories_interventionlifecycles.png", dpi=300, bbox_inches="tight"
)
print("✓ pathway_conceptcategories_interventionlifecycles.png")

# SIMILARITY - Separate figure
print("Generating similarity comparison plot...")
dist_g, dist_c = Counter(lens_g), Counter(lens_c)
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["purple", "blue", "green", "orange", "red"]
if dist_c:
    l, c = zip(*sorted(dist_c.items()))
    ax.loglog(
        l,
        c,
        "s-",
        color=colors[0],
        alpha=0.7,
        label="EDGE only",
        markersize=6,
        linewidth=2,
    )
for idx, t in enumerate(sorted(sim_data.keys())):
    dist = Counter(sim_data[t])
    if dist:
        l, c = zip(*sorted(dist.items()))
        ax.loglog(
            l,
            c,
            "o-",
            color=colors[idx + 1],
            alpha=0.6,
            label=f"EDGE + SIM≥{t}",
            markersize=6,
            linewidth=2,
        )
ax.set_xlabel("Path Length (hops)", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Paths", fontsize=14, fontweight="bold")
ax.set_title(
    "Path Length Distribution: Effect of Similarity Edges\n(Mature interventions, confidence≥3)",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=11, loc="best")
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("similarity_comparison.png", dpi=300, bbox_inches="tight")
print("✓ similarity_comparison.png")

# Comprehensive pathway node degree distributions
print("Generating pathway node degree distributions...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Get degree data from cache
degrees_g = cache["degrees_g"]
degrees_c = cache["degrees_c"]
sim_degrees_c = cache.get("sim_degrees_c", {})

# Colors and markers
gray_color = "#7F7F7F"
colors_c = ["#9B59B6", "#3498DB", "#2ECC71", "#F39C12", "#E74C3C"]

# Log-log plot
# Gray EDGE-only
deg_g = Counter(degrees_g)
if deg_g:
    d, c = zip(*sorted(deg_g.items()))
    ax1.loglog(
        d,
        c,
        "o-",
        color=gray_color,
        alpha=0.6,
        label="EDGE-only (all)",
        markersize=5,
        linewidth=1.5,
    )

# Color (mature) - EDGE-only and similarity
deg_c = Counter(degrees_c)
if deg_c:
    d, c = zip(*sorted(deg_c.items()))
    ax1.loglog(
        d,
        c,
        "s-",
        color=colors_c[0],
        alpha=0.7,
        label="EDGE-only (mature)",
        markersize=6,
        linewidth=2,
    )

for i, t in enumerate([0.80, 0.85, 0.90, 0.95]):
    if t in sim_degrees_c:
        deg = Counter(sim_degrees_c[t])
        if deg:
            d, c = zip(*sorted(deg.items()))
            ax1.loglog(
                d,
                c,
                "s-",
                color=colors_c[i + 1],
                alpha=0.7,
                label=f"EDGE+SIM≥{t} (mature)",
                markersize=5,
                linewidth=1.5,
            )

ax1.set_xlabel("Degree", fontsize=12, fontweight="bold")
ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
ax1.set_title("Pathway Node Degree Distributions", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9, loc="best")
ax1.grid(True, alpha=0.3, which="both")

# CCDF
# Gray EDGE-only
if degrees_g:
    sorted_d = np.sort([d for d in degrees_g if d > 0])
    if len(sorted_d) > 0:
        ccdf = 1 - np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.loglog(
            sorted_d,
            ccdf,
            "o-",
            color=gray_color,
            alpha=0.6,
            markersize=5,
            linewidth=1.5,
            label="EDGE-only (all)",
        )

# Color (mature)
if degrees_c:
    sorted_d = np.sort([d for d in degrees_c if d > 0])
    if len(sorted_d) > 0:
        ccdf = 1 - np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.loglog(
            sorted_d,
            ccdf,
            "s-",
            color=colors_c[0],
            alpha=0.7,
            markersize=6,
            linewidth=2,
            label="EDGE-only (mature)",
        )

for i, t in enumerate([0.80, 0.85, 0.90, 0.95]):
    if t in sim_degrees_c:
        sorted_d = np.sort([d for d in sim_degrees_c[t] if d > 0])
        if len(sorted_d) > 0:
            ccdf = 1 - np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax2.loglog(
                sorted_d,
                ccdf,
                "s-",
                color=colors_c[i + 1],
                alpha=0.7,
                markersize=5,
                linewidth=1.5,
                label=f"EDGE+SIM≥{t} (mature)",
            )

ax2.set_xlabel("Degree", fontsize=12, fontweight="bold")
ax2.set_ylabel("P(Degree ≥ k)", fontsize=12, fontweight="bold")
ax2.set_title("CCDF: Degree Distribution", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, loc="best")
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("pathway_node_degrees_all_EDGEandSIM.png", dpi=300, bbox_inches="tight")
print("✓ pathway_node_degrees_all_EDGEandSIM.png")

# Degree distributions by node type
if "degrees_risks_c" in cache and "sim_degrees_risks_c" in cache:
    print("Generating degree distributions by node type...")

    colors_c = ["#9B59B6", "#3498DB", "#2ECC71", "#F39C12", "#E74C3C"]
    node_types = [
        (
            "risks",
            "Risk",
            cache.get("degrees_risks_c", []),
            cache.get("sim_degrees_risks_c", {}),
        ),
        (
            "interventions",
            "Intervention",
            cache.get("degrees_int_c", []),
            cache.get("sim_degrees_int_c", {}),
        ),
        (
            "other",
            "Other Concept",
            cache.get("degrees_other_c", []),
            cache.get("sim_degrees_other_c", {}),
        ),
    ]

    for type_key, type_label, deg_edge, deg_sim in node_types:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Log-log
        if deg_edge:
            deg_c = Counter(deg_edge)
            d, c = zip(*sorted(deg_c.items()))
            ax1.loglog(
                d,
                c,
                "s-",
                color=colors_c[0],
                alpha=0.7,
                label="EDGE-only",
                markersize=6,
                linewidth=2,
            )

        for i, t in enumerate([0.80, 0.85, 0.90, 0.95]):
            if t in deg_sim and deg_sim[t]:
                deg = Counter(deg_sim[t])
                d, c = zip(*sorted(deg.items()))
                ax1.loglog(
                    d,
                    c,
                    "s-",
                    color=colors_c[i + 1],
                    alpha=0.7,
                    label=f"EDGE+SIM≥{t}",
                    markersize=5,
                    linewidth=1.5,
                )

        ax1.set_xlabel("Degree", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"{type_label} Node Degree Distributions", fontsize=13, fontweight="bold"
        )
        ax1.legend(fontsize=9, loc="best")
        ax1.grid(True, alpha=0.3, which="both")

        # CCDF
        if deg_edge:
            sorted_d = np.sort([d for d in deg_edge if d > 0])
            if len(sorted_d) > 0:
                ccdf = 1 - np.arange(1, len(sorted_d) + 1) / len(sorted_d)
                ax2.loglog(
                    sorted_d,
                    ccdf,
                    "s-",
                    color=colors_c[0],
                    alpha=0.7,
                    markersize=6,
                    linewidth=2,
                    label="EDGE-only",
                )

        for i, t in enumerate([0.80, 0.85, 0.90, 0.95]):
            if t in deg_sim and deg_sim[t]:
                sorted_d = np.sort([d for d in deg_sim[t] if d > 0])
                if len(sorted_d) > 0:
                    ccdf = 1 - np.arange(1, len(sorted_d) + 1) / len(sorted_d)
                    ax2.loglog(
                        sorted_d,
                        ccdf,
                        "s-",
                        color=colors_c[i + 1],
                        alpha=0.7,
                        markersize=5,
                        linewidth=1.5,
                        label=f"EDGE+SIM≥{t}",
                    )

        ax2.set_xlabel("Degree", fontsize=12, fontweight="bold")
        ax2.set_ylabel("P(Degree ≥ k)", fontsize=12, fontweight="bold")
        ax2.set_title(f"{type_label} CCDF", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9, loc="best")
        ax2.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(
            f"pathway_node_degrees_{type_key}_EDGEandSIM.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"✓ pathway_node_degrees_{type_key}_EDGEandSIM.png")

# Per-threshold heatmaps
print("Generating per-threshold category heatmaps...")
bins = list(range(1, 21)) + [">20"]

# EDGE only
fig, ax = plt.subplots(figsize=(10, 6))
mat = np.zeros((len(bins), len(CAT_ORDER)))
for c in comp_c:
    idx = min(c["length"] - 1, len(bins) - 1)
    for cat, cnt in c["categories"].items():
        if cat in CAT_ORDER:
            mat[idx, CAT_ORDER.index(cat)] += cnt
counts = [sum(1 for c in comp_c if c["length"] == i + 1) for i in range(20)]
counts.append(sum(1 for c in comp_c if c["length"] > 20))
for i, cnt in enumerate(counts):
    if cnt > 0:
        mat[i, :] /= cnt
im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=5)
ax.set_xticks(np.arange(len(CAT_ORDER)))
ax.set_xticklabels([c.replace(" ", "\n") for c in CAT_ORDER], fontsize=9)
ax.set_yticks(np.arange(len(bins)))
ax.set_yticklabels(bins, fontsize=9)
ax.set_xlabel("Category", fontsize=11, fontweight="bold")
ax.set_ylabel("Path Length (hops)", fontsize=11, fontweight="bold")
ax.set_title(
    "EDGE Only: Avg Category Frequency per Path Length", fontsize=12, fontweight="bold"
)
plt.colorbar(im, ax=ax, label="Avg/path", ticks=[0, 1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig("heatmap_edge_only.png", dpi=300, bbox_inches="tight")
print("✓ heatmap_edge_only.png")

# Need to extract compositions for similarity thresholds
# This requires reprocessing from cache - skip if sim_data doesn't have compositions
if "sim_comps" in cache:
    for t in sorted(cache["sim_comps"].keys()):
        fig, ax = plt.subplots(figsize=(10, 6))
        mat = np.zeros((len(bins), len(CAT_ORDER)))
        comp_t = cache["sim_comps"][t]

        for c in comp_t:
            idx = min(c["length"] - 1, len(bins) - 1)
            for cat, cnt in c["categories"].items():
                if cat in CAT_ORDER:
                    mat[idx, CAT_ORDER.index(cat)] += cnt

        counts = [sum(1 for c in comp_t if c["length"] == i + 1) for i in range(20)]
        counts.append(sum(1 for c in comp_t if c["length"] > 20))
        for i, cnt in enumerate(counts):
            if cnt > 0:
                mat[i, :] /= cnt

        im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=5)
        ax.set_xticks(np.arange(len(CAT_ORDER)))
        ax.set_xticklabels([c.replace(" ", "\n") for c in CAT_ORDER], fontsize=9)
        ax.set_yticks(np.arange(len(bins)))
        ax.set_yticklabels(bins, fontsize=9)
        ax.set_xlabel("Category", fontsize=11, fontweight="bold")
        ax.set_ylabel("Path Length (hops)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"SIM≥{t}: Avg Category Frequency per Path Length",
            fontsize=12,
            fontweight="bold",
        )
        plt.colorbar(im, ax=ax, label="Avg/path", ticks=[0, 1, 2, 3, 4, 5])
        plt.tight_layout()
        plt.savefig(f"heatmap_sim_{t}.png", dpi=300, bbox_inches="tight")
        print(f"✓ heatmap_sim_{t}.png")
else:
    print("  (Skipping per-threshold heatmaps - compositions not in cache)")

print("Generating per-threshold category heatmaps...")
bins = list(range(1, 21)) + [">20"]

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print("=" * 80)
