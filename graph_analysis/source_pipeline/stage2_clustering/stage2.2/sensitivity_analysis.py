#!/usr/bin/env python3
"""
Sensitivity analysis for paper parameters:
1. α (alpha) - PageRank structural vs similarity weight
2. τ (tau) - Similarity threshold

Output: Tables and metrics for Appendix
"""

import pickle
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from collections import defaultdict
import json
from pathlib import Path

# =============================================================================
# Experiment 1: Alpha Sensitivity (PageRank)
# =============================================================================

def compute_pagerank_for_alpha(G, alpha, beta=0.85):
    """Compute importance PageRank with given alpha"""
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build transition matrices
    struct_rows, struct_cols, struct_vals = [], [], []
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

    # Risk indices for teleportation
    risk_indices = []
    for node in nodes:
        idx = node_to_idx[node]
        attrs = G.nodes[node]
        if attrs.get("type") == "concept" and attrs.get("concept_category") == "risk":
            risk_indices.append(idx)

    # Personalized PageRank
    v = np.zeros(n)
    v[risk_indices] = 1.0 / len(risk_indices)
    r = np.ones(n) / n

    for _ in range(100):
        r_new = beta * (P.T @ r) + (1 - beta) * v
        if np.linalg.norm(r_new - r, 1) < 1e-6:
            break
        r = r_new

    return {nodes[i]: r[i] for i in range(n)}


def experiment_alpha_sensitivity(G, alphas=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    """Test PageRank sensitivity to alpha parameter"""
    print("\n" + "="*60)
    print("Experiment 1: Alpha Sensitivity Analysis")
    print("="*60)

    results = {}
    rankings = {}

    for alpha in alphas:
        print(f"\nComputing PageRank with α={alpha}...")
        pr = compute_pagerank_for_alpha(G, alpha)
        results[alpha] = pr

        # Get risk nodes and their rankings
        risk_pr = {n: v for n, v in pr.items()
                   if G.nodes[n].get("type") == "concept"
                   and G.nodes[n].get("concept_category") == "risk"}

        sorted_risks = sorted(risk_pr.items(), key=lambda x: x[1], reverse=True)
        rankings[alpha] = [n for n, _ in sorted_risks]

    # Compute correlations between rankings
    print("\n" + "-"*60)
    print("Spearman Rank Correlations between α values:")
    print("-"*60)

    # Create ranking vectors for correlation
    base_alpha = 0.7  # Our chosen value
    base_ranking = rankings[base_alpha]
    base_rank_dict = {n: i for i, n in enumerate(base_ranking)}

    correlation_results = {}
    for alpha in alphas:
        other_ranking = rankings[alpha]
        # Get ranks for same nodes
        base_ranks = []
        other_ranks = []
        for i, n in enumerate(other_ranking):
            if n in base_rank_dict:
                base_ranks.append(base_rank_dict[n])
                other_ranks.append(i)

        corr, pval = spearmanr(base_ranks, other_ranks)
        correlation_results[alpha] = {"correlation": corr, "p_value": pval}
        print(f"  α={base_alpha} vs α={alpha}: ρ={corr:.4f} (p={pval:.2e})")

    # Top-20 overlap
    print("\n" + "-"*60)
    print("Top-20 Risk Overlap (Jaccard) vs α=0.7:")
    print("-"*60)

    top20_base = set(rankings[base_alpha][:20])
    overlap_results = {}
    for alpha in alphas:
        top20_other = set(rankings[alpha][:20])
        jaccard = len(top20_base & top20_other) / len(top20_base | top20_other)
        overlap_results[alpha] = jaccard
        overlap_count = len(top20_base & top20_other)
        print(f"  α={alpha}: {overlap_count}/20 overlap (Jaccard={jaccard:.3f})")

    return {
        "correlations": correlation_results,
        "top20_overlap": overlap_results,
        "rankings": {a: r[:50] for a, r in rankings.items()}  # Save top 50 for inspection
    }


# =============================================================================
# Experiment 2: Tau Sensitivity (Similarity Threshold)
# =============================================================================

def compute_similarity_stats(G, embeddings_by_category, tau):
    """Compute stats for a given similarity threshold"""

    try:
        import faiss
        HAS_FAISS = True
    except ImportError:
        HAS_FAISS = False

    total_edges = 0

    for category, node_embs in embeddings_by_category.items():
        if len(node_embs) < 2:
            continue

        node_ids = [n for n, _ in node_embs]
        embeddings = np.array([e for _, e in node_embs], dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        if HAS_FAISS and len(node_ids) > 100:
            # Use FAISS
            index = faiss.IndexFlatIP(embeddings_norm.shape[1])
            index.add(embeddings_norm)
            k = min(100, len(node_ids))
            similarities, indices = index.search(embeddings_norm, k)

            for i in range(len(node_ids)):
                for j_pos in range(k):
                    j = indices[i, j_pos]
                    if j > i and similarities[i, j_pos] >= tau:
                        total_edges += 1
        else:
            # NumPy fallback
            sim_matrix = embeddings_norm @ embeddings_norm.T
            rows, cols = np.triu_indices(len(node_ids), k=1)
            pairs_above = np.sum(sim_matrix[rows, cols] >= tau)
            total_edges += pairs_above

    return total_edges


def experiment_tau_sensitivity(base_graph_path, taus=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95]):
    """Test similarity threshold sensitivity"""
    print("\n" + "="*60)
    print("Experiment 2: Tau (Similarity Threshold) Sensitivity")
    print("="*60)

    # Load base graph (before similarity edges)
    print(f"\nLoading base graph from {base_graph_path}...")
    with open(base_graph_path, 'rb') as f:
        G_base = pickle.load(f)

    print(f"Base graph: {G_base.number_of_nodes():,} nodes, {G_base.number_of_edges():,} edges")

    # Extract embeddings by category
    CATEGORIES_WITH_SIMILARITY = [
        "design rationale",
        "implementation mechanism",
        "problem analysis",
        "theoretical insight",
        "validation evidence",
    ]

    embeddings_by_category = defaultdict(list)
    interventions = []

    for node_id, attrs in G_base.nodes(data=True):
        emb = attrs.get("embedding")
        if emb is None:
            continue
        emb = np.array(emb, dtype=np.float32)

        if attrs.get("type") == "concept":
            category = attrs.get("concept_category", "unknown")
            if category in CATEGORIES_WITH_SIMILARITY:
                embeddings_by_category[category].append((node_id, emb))
        elif attrs.get("type") == "intervention":
            interventions.append((node_id, emb))

    # Add interventions
    embeddings_by_category["intervention"] = interventions

    print("\nEmbeddings by category:")
    for cat, nodes in sorted(embeddings_by_category.items()):
        print(f"  {cat}: {len(nodes):,}")

    # Compute stats for each threshold
    print("\n" + "-"*60)
    print("Similarity edges by threshold:")
    print("-"*60)

    results = {}
    for tau in taus:
        print(f"\nComputing for τ={tau}...")
        edges = compute_similarity_stats(G_base, embeddings_by_category, tau)
        results[tau] = {"similarity_edges": edges}
        print(f"  τ={tau}: {edges:,} similarity edges")

    # Print summary table
    print("\n" + "-"*60)
    print("Summary Table:")
    print("-"*60)
    print(f"{'τ':>6} | {'Sim Edges':>12} | {'Ratio vs 0.80':>14}")
    print("-"*6 + "-+-" + "-"*12 + "-+-" + "-"*14)

    base_edges = results[0.80]["similarity_edges"]
    for tau in taus:
        edges = results[tau]["similarity_edges"]
        ratio = edges / base_edges if base_edges > 0 else 0
        marker = " *" if tau == 0.80 else ""
        print(f"{tau:>6.2f} | {edges:>12,} | {ratio:>13.2f}x{marker}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parameter sensitivity analysis")
    parser.add_argument("--graph-with-sim", type=str,
                       default="results/graph_stage2.2_with_similarity.pkl",
                       help="Graph with similarity edges (for alpha experiment)")
    parser.add_argument("--graph-base", type=str,
                       default="../stage2.1/results/k_40_umap/step2_models/graph_stage2_umap_k40.pkl",
                       help="Base graph without similarity edges (for tau experiment)")
    parser.add_argument("--output", type=str, default="results/sensitivity_analysis.json",
                       help="Output JSON file")
    parser.add_argument("--alpha-only", action="store_true", help="Run only alpha experiment")
    parser.add_argument("--tau-only", action="store_true", help="Run only tau experiment")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    results = {}

    # Experiment 1: Alpha
    if not args.tau_only:
        graph_path = script_dir / args.graph_with_sim
        print(f"\nLoading graph with similarity edges from {graph_path}...")
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        print(f"Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

        results["alpha"] = experiment_alpha_sensitivity(G)

    # Experiment 2: Tau
    if not args.alpha_only:
        base_path = script_dir / args.graph_base
        results["tau"] = experiment_tau_sensitivity(base_path)

    # Save results
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results_json = convert_numpy(results)

    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
