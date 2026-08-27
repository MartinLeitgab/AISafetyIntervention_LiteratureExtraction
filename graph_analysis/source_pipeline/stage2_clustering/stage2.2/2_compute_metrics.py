#!/usr/bin/env python3
"""
Stage 2.2: Compute graph metrics

Metrics computed:
1. Degree metrics (all nodes): in_degree_struct, out_degree_struct, sim_degree
2. Centralities (all nodes): betweenness_centrality (sampled), eigenvector_centrality
3. PageRank (all nodes): pagerank_importance, pagerank_research
4. Coverage for risks: reached_intervention, max_forward_category, evidence_count, intervention_count, source_count
5. Understudied score (risks only): pagerank_importance / (evidence_count + 1)
6. Coverage for interventions: risk_coverage
7. Path template statistics (aggregated)
"""

import pickle
import json
import numpy as np
from collections import defaultdict, deque, Counter
from pathlib import Path
import argparse
import logging
import time
import networkx as nx
from scipy import sparse

# Category ordering for path analysis
CATEGORY_ORDER = {
    "risk": 0,
    "problem analysis": 1,
    "theoretical insight": 2,
    "design rationale": 3,
    "implementation mechanism": 4,
    "validation evidence": 5,
    "intervention": 6,
}


def compute_path_forward_progress(path, G):
    """
    Compute max forward category index for a path.

    Rules:
    - Order matters: risk(0) → problem(1) → ... → intervention(6)
    - Repetitions within same category OK
    - Backtrack = stop (return current max)
    - Intervention = stop (return 6)

    Returns: 0-6 (max category index reached without backtracking)
    """
    if not path:
        return 0

    max_idx = 0  # start at risk
    for node in path[1:]:  # skip starting risk
        cat = get_node_category(G, node)
        idx = CATEGORY_ORDER.get(cat, -1)

        if idx == 6:  # intervention reached
            return 6
        elif idx >= max_idx:  # forward or same category
            max_idx = idx
        else:  # backtrack detected
            return max_idx

    return max_idx


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w', encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_graph(path):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading graph from {path}...")
    with open(path, 'rb') as f:
        G = pickle.load(f)
    logger.info(f"Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def get_node_category(G, node):
    """Get category for a node (concept_category or 'intervention')"""
    attrs = G.nodes[node]
    if attrs.get("type") == "intervention":
        return "intervention"
    return attrs.get("concept_category", "unknown")


def get_struct_neighbors(G, node):
    """Get neighbors connected by structural edges"""
    neighbors = []
    for neighbor in G.neighbors(node):
        edge_data = G.edges[node, neighbor]
        if edge_data.get("type") != "SIMILAR":
            neighbors.append(neighbor)
    # For directed graphs, also check predecessors
    if G.is_directed():
        for neighbor in G.predecessors(node):
            edge_data = G.edges[neighbor, node]
            if edge_data.get("type") != "SIMILAR":
                neighbors.append(neighbor)
    return neighbors


def get_sim_neighbors(G, node):
    """Get neighbors connected by SIMILAR edges"""
    neighbors = []
    for neighbor in G.neighbors(node):
        edge_data = G.edges[node, neighbor]
        if edge_data.get("type") == "SIMILAR":
            neighbors.append(neighbor)
    if G.is_directed():
        for neighbor in G.predecessors(node):
            edge_data = G.edges[neighbor, node]
            if edge_data.get("type") == "SIMILAR":
                neighbors.append(neighbor)
    return neighbors


# =============================================================================
# 1. Degree Metrics
# =============================================================================

def compute_degree_metrics(G):
    """Compute degree metrics for all nodes"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("Computing degree metrics...")
    logger.info("="*60)

    start = time.time()

    for node in G.nodes():
        in_struct = 0
        out_struct = 0
        sim_deg = 0

        # Outgoing edges
        for neighbor in G.neighbors(node):
            edge_data = G.edges[node, neighbor]
            if edge_data.get("type") == "SIMILAR":
                sim_deg += 1
            else:
                out_struct += 1

        # Incoming edges (for directed graph)
        if G.is_directed():
            for neighbor in G.predecessors(node):
                edge_data = G.edges[neighbor, node]
                if edge_data.get("type") == "SIMILAR":
                    sim_deg += 1
                else:
                    in_struct += 1

        G.nodes[node]["in_degree_struct"] = in_struct
        G.nodes[node]["out_degree_struct"] = out_struct
        G.nodes[node]["sim_degree"] = sim_deg

    elapsed = time.time() - start
    logger.info(f"Degree metrics computed in {elapsed:.1f}s")

    return G


# =============================================================================
# 2. Centralities
# =============================================================================

def compute_centralities(G, betweenness_k=5000):
    """Compute centrality metrics"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("Computing centralities...")
    logger.info("="*60)

    # Convert to undirected for centrality computation
    G_undirected = G.to_undirected()

    # Betweenness centrality (sampled)
    logger.info(f"Computing betweenness centrality (k={betweenness_k} samples)...")
    start = time.time()
    betweenness = nx.betweenness_centrality(G_undirected, k=betweenness_k)
    elapsed = time.time() - start
    logger.info(f"Betweenness computed in {elapsed:.1f}s")

    for node, value in betweenness.items():
        G.nodes[node]["betweenness_centrality"] = value

    # Eigenvector centrality
    logger.info("Computing eigenvector centrality...")
    start = time.time()
    try:
        eigenvector = nx.eigenvector_centrality(G_undirected, max_iter=1000)
        for node, value in eigenvector.items():
            G.nodes[node]["eigenvector_centrality"] = value
        elapsed = time.time() - start
        logger.info(f"Eigenvector centrality computed in {elapsed:.1f}s")
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality failed to converge, using degree centrality as fallback")
        degree_cent = nx.degree_centrality(G_undirected)
        for node, value in degree_cent.items():
            G.nodes[node]["eigenvector_centrality"] = value

    return G


# =============================================================================
# 3. PageRank Metrics
# =============================================================================

def compute_pagerank_metrics(G, alpha=0.7, beta=0.85):
    """
    Compute custom PageRank metrics:
    - pagerank_importance: teleport to risk nodes
    - pagerank_research: teleport to evidence + mature interventions
    - understudied_score: importance / research (for risks only)
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("Computing PageRank metrics...")
    logger.info(f"Parameters: α={alpha} (struct weight), β={beta} (damping)")
    logger.info("="*60)

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build transition matrices
    logger.info("Building transition matrices...")
    start = time.time()

    # Structural edges matrix
    struct_rows, struct_cols, struct_vals = [], [], []
    # Similarity edges matrix
    sim_rows, sim_cols, sim_vals = [], [], []

    for u, v, data in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        if data.get("type") == "SIMILAR":
            sim_rows.extend([i, j])
            sim_cols.extend([j, i])
            sim_vals.extend([1.0, 1.0])
        else:
            struct_rows.append(i)
            struct_cols.append(j)
            struct_vals.append(1.0)

    P_struct = sparse.csr_matrix((struct_vals, (struct_rows, struct_cols)), shape=(n, n))
    P_sim = sparse.csr_matrix((sim_vals, (sim_rows, sim_cols)), shape=(n, n))

    # Row-normalize
    def row_normalize(M):
        row_sums = np.array(M.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        return sparse.diags(1.0 / row_sums) @ M

    P_struct = row_normalize(P_struct)
    P_sim = row_normalize(P_sim)

    # Combined transition matrix
    P = alpha * P_struct + (1 - alpha) * P_sim

    elapsed = time.time() - start
    logger.info(f"Transition matrices built in {elapsed:.1f}s")

    # Identify node sets
    risk_indices = []
    evidence_indices = []
    mature_intervention_indices = []

    for node in nodes:
        idx = node_to_idx[node]
        attrs = G.nodes[node]

        if attrs.get("type") == "concept" and attrs.get("concept_category") == "risk":
            risk_indices.append(idx)
        elif attrs.get("type") == "concept" and attrs.get("concept_category") == "validation evidence":
            evidence_indices.append(idx)
        elif attrs.get("type") == "intervention":
            maturity = attrs.get("maturity", 0)
            if maturity >= 3:
                mature_intervention_indices.append(idx)

    research_indices = evidence_indices + mature_intervention_indices

    logger.info(f"Risk nodes: {len(risk_indices):,}")
    logger.info(f"Research nodes (evidence + mature interventions): {len(research_indices):,}")

    def personalized_pagerank(P, personalization_indices, beta, max_iter=100, tol=1e-6):
        """Compute personalized PageRank with custom teleport"""
        n = P.shape[0]

        # Teleport vector
        v = np.zeros(n)
        if personalization_indices:
            v[personalization_indices] = 1.0 / len(personalization_indices)
        else:
            v = np.ones(n) / n

        # Initialize
        r = np.ones(n) / n

        # Power iteration
        for _ in range(max_iter):
            r_new = beta * (P.T @ r) + (1 - beta) * v
            if np.linalg.norm(r_new - r, 1) < tol:
                break
            r = r_new

        return r

    # Importance PageRank (teleport to risks)
    logger.info("Computing importance PageRank (teleport to risks)...")
    start = time.time()
    pr_importance = personalized_pagerank(P, risk_indices, beta)
    elapsed = time.time() - start
    logger.info(f"Importance PageRank computed in {elapsed:.1f}s")

    # Research PageRank (teleport to evidence + mature interventions)
    logger.info("Computing research PageRank (teleport to evidence + mature interventions)...")
    start = time.time()
    pr_research = personalized_pagerank(P, research_indices, beta)
    elapsed = time.time() - start
    logger.info(f"Research PageRank computed in {elapsed:.1f}s")

    # Store results
    for node in nodes:
        idx = node_to_idx[node]
        G.nodes[node]["pagerank_importance"] = float(pr_importance[idx])
        G.nodes[node]["pagerank_research"] = float(pr_research[idx])

    return G


# =============================================================================
# 4. Coverage Metrics for Risks (BFS with sim hop)
# =============================================================================

def bfs_with_sim_hop(G, start_node, max_sim_hop=2, track_paths=False):
    """
    BFS with limited similarity hops.
    First hop from start_node must be structural.

    Returns:
        scope: set of reachable nodes
        paths: dict of node -> path (if track_paths=True)
    """
    queue = deque([(start_node, 0, False, [start_node])])  # (node, sim_hops, has_struct, path)
    visited = {start_node}
    scope = {start_node}
    paths = {start_node: [start_node]} if track_paths else None

    while queue:
        node, sim_hops, has_struct, path = queue.popleft()

        # Structural neighbors (always allowed)
        for neighbor in get_struct_neighbors(G, node):
            if neighbor not in visited:
                visited.add(neighbor)
                scope.add(neighbor)
                new_path = path + [neighbor] if track_paths else []
                if track_paths:
                    paths[neighbor] = new_path
                queue.append((neighbor, sim_hops, True, new_path))

        # Similarity neighbors (only after struct hop and if sim_hops < max)
        if has_struct and sim_hops < max_sim_hop:
            for neighbor in get_sim_neighbors(G, node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    scope.add(neighbor)
                    new_path = path + [neighbor] if track_paths else []
                    if track_paths:
                        paths[neighbor] = new_path
                    queue.append((neighbor, sim_hops + 1, True, new_path))

    return scope, paths


def compute_risk_coverage(G, max_sim_hop=2):
    """Compute coverage metrics for risk nodes"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("Computing coverage metrics for risks...")
    logger.info(f"Parameters: max_sim_hop={max_sim_hop}")
    logger.info("="*60)

    start = time.time()

    # Find all risk nodes
    risk_nodes = [n for n in G.nodes() if get_node_category(G, n) == "risk"]
    logger.info(f"Processing {len(risk_nodes):,} risk nodes...")

    for i, risk_node in enumerate(risk_nodes):
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1:,}/{len(risk_nodes):,} risks...")

        scope, paths = bfs_with_sim_hop(G, risk_node, max_sim_hop, track_paths=True)

        # Count by category
        evidence_count = 0
        intervention_count = 0
        sources = set()
        reached_intervention = False

        for node in scope:
            cat = get_node_category(G, node)

            if cat == "validation evidence":
                evidence_count += 1
            elif cat == "intervention":
                intervention_count += 1
                reached_intervention = True

            # Collect sources
            source_url = G.nodes[node].get("source_url")
            if source_url:
                sources.add(source_url)

        # Max forward category (order-aware, stops at intervention or backtrack)
        max_forward = 0
        if reached_intervention:
            # Check paths to interventions
            for node in scope:
                if get_node_category(G, node) == "intervention" and node in paths:
                    path = paths[node]
                    forward = compute_path_forward_progress(path, G)
                    max_forward = max(max_forward, forward)
        else:
            # Check all paths
            for node, path in paths.items():
                forward = compute_path_forward_progress(path, G)
                max_forward = max(max_forward, forward)

        # Store metrics
        G.nodes[risk_node]["reached_intervention"] = reached_intervention
        G.nodes[risk_node]["max_forward_category"] = max_forward
        G.nodes[risk_node]["evidence_count"] = evidence_count
        G.nodes[risk_node]["intervention_count"] = intervention_count
        G.nodes[risk_node]["source_count"] = len(sources)

    elapsed = time.time() - start
    logger.info(f"Risk coverage computed in {elapsed:.1f}s")

    # Summary stats
    reached = sum(1 for n in risk_nodes if G.nodes[n].get("reached_intervention", False))
    logger.info(f"Risks reaching intervention: {reached:,}/{len(risk_nodes):,} ({100*reached/len(risk_nodes):.1f}%)")

    return G


# =============================================================================
# 4.5. Understudied Score (requires evidence_count from risk coverage)
# =============================================================================

def compute_understudied_score(G):
    """
    Compute understudied_score for risks.
    Formula: pagerank_importance / (evidence_count + 1)

    Rationale: High importance (structurally central) + low evidence = understudied
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("Computing understudied_score...")
    logger.info("Formula: pagerank_importance / (evidence_count + 1)")
    logger.info("="*60)

    risk_nodes = [n for n in G.nodes() if get_node_category(G, n) == "risk"]

    for node in risk_nodes:
        importance = G.nodes[node].get("pagerank_importance", 0)
        evidence_count = G.nodes[node].get("evidence_count", 0)

        G.nodes[node]["understudied_score"] = float(importance / (evidence_count + 1))

    logger.info(f"Understudied score computed for {len(risk_nodes):,} risks")

    return G


# =============================================================================
# 5. Coverage Metrics for Interventions
# =============================================================================

def compute_intervention_coverage(G, max_sim_hop=2):
    """Compute risk_coverage for intervention nodes (reverse BFS)"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("Computing coverage metrics for interventions...")
    logger.info(f"Parameters: max_sim_hop={max_sim_hop}")
    logger.info("="*60)

    start = time.time()

    # Find all intervention nodes
    intervention_nodes = [n for n in G.nodes() if get_node_category(G, n) == "intervention"]
    logger.info(f"Processing {len(intervention_nodes):,} intervention nodes...")

    for i, intervention_node in enumerate(intervention_nodes):
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1:,}/{len(intervention_nodes):,} interventions...")

        scope, _ = bfs_with_sim_hop(G, intervention_node, max_sim_hop, track_paths=False)

        # Count risks in scope
        risk_count = sum(1 for n in scope if get_node_category(G, n) == "risk")

        G.nodes[intervention_node]["risk_coverage"] = risk_count

    elapsed = time.time() - start
    logger.info(f"Intervention coverage computed in {elapsed:.1f}s")

    return G


# =============================================================================
# 6. Path Template Statistics
# =============================================================================

def compute_path_statistics(G, max_sim_hop=2):
    """Compute path template statistics (aggregated)"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("Computing path template statistics...")
    logger.info("="*60)

    start = time.time()

    # Find all risk nodes
    risk_nodes = [n for n in G.nodes() if get_node_category(G, n) == "risk"]

    complete_patterns = Counter()  # Patterns reaching intervention
    partial_patterns = Counter()   # Patterns not reaching intervention

    for i, risk_node in enumerate(risk_nodes):
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1:,}/{len(risk_nodes):,} risks...")

        scope, paths = bfs_with_sim_hop(G, risk_node, max_sim_hop, track_paths=True)

        # Find paths to interventions
        intervention_paths = []
        for node in scope:
            if get_node_category(G, node) == "intervention" and node in paths:
                intervention_paths.append(paths[node])

        if intervention_paths:
            # Complete patterns
            for path in intervention_paths:
                pattern = tuple(get_node_category(G, n) for n in path)
                complete_patterns[pattern] += 1
        else:
            # Partial patterns (all terminal paths)
            terminal_nodes = [n for n in scope if n not in [start_node for start_node, _, _, _ in []]]
            for node, path in paths.items():
                # Check if this is a terminal node (no unvisited neighbors)
                pattern = tuple(get_node_category(G, n) for n in path)
                if len(path) > 1:  # Skip start node only
                    partial_patterns[pattern] += 1

    elapsed = time.time() - start
    logger.info(f"Path statistics computed in {elapsed:.1f}s")

    # Format results
    stats = {
        "complete_patterns": [
            {"pattern": list(p), "count": c}
            for p, c in complete_patterns.most_common(50)
        ],
        "partial_patterns": [
            {"pattern": list(p), "count": c}
            for p, c in partial_patterns.most_common(50)
        ],
        "total_complete": sum(complete_patterns.values()),
        "total_partial": sum(partial_patterns.values()),
        "unique_complete_patterns": len(complete_patterns),
        "unique_partial_patterns": len(partial_patterns),
    }

    logger.info(f"Complete patterns: {stats['total_complete']:,} total, {stats['unique_complete_patterns']} unique")
    logger.info(f"Partial patterns: {stats['total_partial']:,} total, {stats['unique_partial_patterns']} unique")

    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute graph metrics')
    parser.add_argument('--input', required=True, help='Input graph path')
    parser.add_argument('--output', required=True, help='Output graph path')
    parser.add_argument('--log', type=str, help='Log file path')
    parser.add_argument('--betweenness-k', type=int, default=5000, help='Betweenness centrality samples')
    parser.add_argument('--max-sim-hop', type=int, default=2, help='Max similarity hops in BFS')
    parser.add_argument('--alpha', type=float, default=0.7, help='PageRank struct weight')
    parser.add_argument('--beta', type=float, default=0.85, help='PageRank damping factor')

    args = parser.parse_args()

    logger = setup_logging(args.log)

    logger.info("="*60)
    logger.info("STAGE 2.2: Compute Graph Metrics")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Betweenness k: {args.betweenness_k}")
    logger.info(f"Max sim hop: {args.max_sim_hop}")
    logger.info(f"PageRank α: {args.alpha}")
    logger.info(f"PageRank β: {args.beta}")

    total_start = time.time()

    # Load graph
    G = load_graph(args.input)

    # 1. Degree metrics
    G = compute_degree_metrics(G)

    # 2. Centralities
    G = compute_centralities(G, betweenness_k=args.betweenness_k)

    # 3. PageRank
    G = compute_pagerank_metrics(G, alpha=args.alpha, beta=args.beta)

    # 4. Risk coverage
    G = compute_risk_coverage(G, max_sim_hop=args.max_sim_hop)

    # 4.5. Understudied score (requires evidence_count from step 4)
    G = compute_understudied_score(G)

    # 5. Intervention coverage
    G = compute_intervention_coverage(G, max_sim_hop=args.max_sim_hop)

    # 6. Path statistics
    path_stats = compute_path_statistics(G, max_sim_hop=args.max_sim_hop)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving graph to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)

    # Save path statistics
    stats_path = output_path.parent / "path_statistics.json"
    logger.info(f"Saving path statistics to {stats_path}...")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(path_stats, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - total_start

    logger.info(f"\n{'='*60}")
    logger.info("COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Output graph: {output_path}")
    logger.info(f"Path statistics: {stats_path}")


if __name__ == "__main__":
    main()
