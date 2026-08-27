#!/usr/bin/env python3
"""
Stage 2.2: Add similarity edges to graph

Parameters:
- Embedding: 1536D (Wide) - attribute 'embedding'
- Metric: Cosine similarity
- Threshold: >= 0.80
- Scope: Within category (EXCLUDING risk - to avoid echo chambers in PageRank)
- Edge type: SIMILAR
"""

import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse
import logging
from datetime import datetime

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: FAISS not available, using numpy (slower)")

# Categories to process (risk excluded to avoid echo chambers in PageRank)
CATEGORIES_WITH_SIMILARITY = [
    "design rationale",
    "implementation mechanism",
    "problem analysis",
    "theoretical insight",
    "validation evidence",
]


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
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


def extract_embeddings_by_category(G):
    """Extract 1536D embeddings grouped by type and category"""
    logger = logging.getLogger(__name__)
    logger.info("Extracting embeddings by category...")

    concepts_by_category = defaultdict(list)  # category -> [(node_id, embedding)]
    interventions = []
    missing = 0
    skipped_risk = 0

    for node_id, attrs in G.nodes(data=True):
        emb = attrs.get("embedding")

        if emb is None:
            missing += 1
            continue

        emb = np.array(emb, dtype=np.float32)

        if attrs.get("type") == "concept":
            category = attrs.get("concept_category", "unknown")
            if category == "risk":
                skipped_risk += 1
                continue  # Skip risks - no similarity edges for them
            if category in CATEGORIES_WITH_SIMILARITY:
                concepts_by_category[category].append((node_id, emb))
        elif attrs.get("type") == "intervention":
            interventions.append((node_id, emb))

    logger.info(f"Missing embeddings: {missing}")
    logger.info(f"Skipped risk nodes: {skipped_risk}")
    logger.info(f"Concepts by category (for similarity):")
    for cat, nodes in sorted(concepts_by_category.items()):
        logger.info(f"  {cat}: {len(nodes):,}")
    logger.info(f"Interventions: {len(interventions):,}")

    return concepts_by_category, interventions


def find_similar_pairs_faiss(node_embeddings, threshold=0.80, k=100):
    """Use FAISS to find pairs with cosine similarity >= threshold"""

    if len(node_embeddings) < 2:
        return []

    node_ids = [n[0] for n in node_embeddings]
    embeddings = np.array([n[1] for n in node_embeddings], dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms

    # FAISS index for inner product (= cosine for normalized vectors)
    d = embeddings_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_norm)

    # Search k nearest neighbors
    k = min(k, len(node_ids))
    similarities, indices = index.search(embeddings_norm, k)

    # Collect pairs above threshold
    pairs = []
    for i in range(len(node_ids)):
        for j_idx in range(k):
            j = indices[i, j_idx]
            sim = similarities[i, j_idx]

            if j <= i:  # Skip self and already counted pairs
                continue

            if sim >= threshold:
                pairs.append((node_ids[i], node_ids[j], float(sim)))

    return pairs


def find_similar_pairs_numpy(node_embeddings, threshold=0.80):
    """Fallback numpy implementation"""

    if len(node_embeddings) < 2:
        return []

    node_ids = [n[0] for n in node_embeddings]
    embeddings = np.array([n[1] for n in node_embeddings], dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms

    # Compute similarity matrix
    sim_matrix = embeddings_norm @ embeddings_norm.T

    # Find pairs above threshold
    pairs = []
    n = len(node_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                pairs.append((node_ids[i], node_ids[j], float(sim_matrix[i, j])))

    return pairs


def add_similarity_edges(G, threshold=0.80):
    """Add SIMILAR edges to graph based on cosine similarity"""
    logger = logging.getLogger(__name__)

    concepts_by_category, interventions = extract_embeddings_by_category(G)

    find_pairs = find_similar_pairs_faiss if HAS_FAISS else find_similar_pairs_numpy

    total_edges = 0
    edges_by_category = {}

    logger.info(f"\n{'='*60}")
    logger.info(f"Adding SIMILAR edges (threshold >= {threshold})")
    logger.info(f"NOTE: risk↔risk similarity EXCLUDED (avoid PageRank echo chambers)")
    logger.info(f"{'='*60}")

    # Process each concept category (excluding risk)
    for category, nodes in sorted(concepts_by_category.items()):
        logger.info(f"\nProcessing {category} ({len(nodes):,} nodes)...")

        pairs = find_pairs(nodes, threshold)

        for node1, node2, sim in pairs:
            G.add_edge(node1, node2,
                      type="SIMILAR",
                      similarity_score=sim,
                      category=category)

        total_edges += len(pairs)
        edges_by_category[category] = len(pairs)
        logger.info(f"  Added {len(pairs):,} SIMILAR edges")

    # Process interventions
    logger.info(f"\nProcessing interventions ({len(interventions):,} nodes)...")

    pairs = find_pairs(interventions, threshold)

    for node1, node2, sim in pairs:
        G.add_edge(node1, node2,
                  type="SIMILAR",
                  similarity_score=sim,
                  category="intervention")

    total_edges += len(pairs)
    edges_by_category["interventions"] = len(pairs)
    logger.info(f"  Added {len(pairs):,} SIMILAR edges")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total SIMILAR edges added: {total_edges:,}")
    logger.info(f"Total edges in graph: {G.number_of_edges():,}")
    logger.info(f"\nBy category:")
    for cat, count in sorted(edges_by_category.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count:,}")

    return G, {
        "total_similar_edges": total_edges,
        "edges_by_category": edges_by_category
    }


def main():
    parser = argparse.ArgumentParser(description='Add similarity edges to graph')
    parser.add_argument('--input', required=True, help='Input graph path')
    parser.add_argument('--output', required=True, help='Output graph path')
    parser.add_argument('--threshold', type=float, default=0.80, help='Similarity threshold (default: 0.80)')
    parser.add_argument('--log', type=str, help='Log file path')

    args = parser.parse_args()

    logger = setup_logging(args.log)

    logger.info("="*60)
    logger.info("STAGE 2.2: Add Similarity Edges")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"FAISS available: {HAS_FAISS}")
    logger.info(f"Categories with similarity: {CATEGORIES_WITH_SIMILARITY}")
    logger.info(f"risk↔risk: EXCLUDED")

    # Load graph
    G = load_graph(args.input)

    # Add similarity edges
    G, stats = add_similarity_edges(G, threshold=args.threshold)

    # Save graph
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving graph to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)

    logger.info(f"\n{'='*60}")
    logger.info("COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
