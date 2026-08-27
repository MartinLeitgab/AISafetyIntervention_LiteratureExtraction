#!/usr/bin/env python3
"""
Fast tau sensitivity analysis.
Computes similarity scores once, then counts pairs for each threshold.
"""

import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
import sys

def main():
    script_dir = Path(__file__).parent
    base_path = script_dir / "../stage2.1/results/k_40_umap/step2_models/graph_stage2_umap_k40.pkl"

    print(f"Loading base graph from {base_path}...")
    sys.stdout.flush()
    with open(base_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    sys.stdout.flush()

    CATEGORIES = [
        "design rationale",
        "implementation mechanism",
        "problem analysis",
        "theoretical insight",
        "validation evidence",
        "intervention",
    ]

    # Extract embeddings by category
    print("\nExtracting embeddings...")
    sys.stdout.flush()
    embeddings_by_category = defaultdict(list)

    for node_id, attrs in G.nodes(data=True):
        emb = attrs.get("embedding")
        if emb is None:
            continue
        emb = np.array(emb, dtype=np.float32)

        if attrs.get("type") == "concept":
            category = attrs.get("concept_category", "unknown")
            if category in CATEGORIES[:5]:  # Non-intervention concepts
                embeddings_by_category[category].append((node_id, emb))
        elif attrs.get("type") == "intervention":
            embeddings_by_category["intervention"].append((node_id, emb))

    for cat, nodes in sorted(embeddings_by_category.items()):
        print(f"  {cat}: {len(nodes):,}")
    sys.stdout.flush()

    # Compute all similarities and store them
    print("\nComputing similarities (this takes a while)...")
    sys.stdout.flush()

    taus = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    counts = {tau: 0 for tau in taus}

    for category, node_embs in embeddings_by_category.items():
        if len(node_embs) < 2:
            continue

        print(f"\n  Processing {category} ({len(node_embs):,} nodes)...")
        sys.stdout.flush()

        embeddings = np.array([e for _, e in node_embs], dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        n = len(node_embs)

        # Process in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)
            batch = embeddings_norm[i:i_end]

            # Compute similarities to all nodes with index > i
            # to avoid counting pairs twice
            sims = batch @ embeddings_norm[i:].T

            # Only upper triangle (j > i within this computation)
            for bi in range(batch.shape[0]):
                global_i = i + bi
                # Similarities to nodes with index > global_i
                start_j = bi + 1  # skip self and earlier indices
                for tau in taus:
                    counts[tau] += np.sum(sims[bi, start_j:] >= tau)

            if i % 5000 == 0 and i > 0:
                print(f"    Processed {i:,}/{n:,} nodes...")
                sys.stdout.flush()

    # Print results
    print("\n" + "="*60)
    print("Tau Sensitivity Results")
    print("="*60)
    print(f"{'τ':>6} | {'Sim Edges':>12} | {'Ratio vs 0.80':>14}")
    print("-"*6 + "-+-" + "-"*12 + "-+-" + "-"*14)

    base = counts[0.80]
    for tau in taus:
        edges = counts[tau]
        ratio = edges / base if base > 0 else 0
        marker = " *" if tau == 0.80 else ""
        print(f"{tau:>6.2f} | {edges:>12,} | {ratio:>13.2f}x{marker}")

    # Save results
    import json
    results = {
        "tau_sensitivity": {str(tau): count for tau, count in counts.items()}
    }
    output_path = script_dir / "results/tau_sensitivity.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
