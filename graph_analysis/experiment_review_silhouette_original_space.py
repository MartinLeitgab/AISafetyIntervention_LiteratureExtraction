#!/usr/bin/env python
"""Is the UMAP silhouette gain real, or an artifact of scoring clusters in UMAP space?

Reviewer item W-6 / Q-W6 (both simulated reviews, 2026-08-14): the manuscript contrasts a
silhouette of 0.022 for k-means in the original 1536-dimensional embedding space against
0.298 for k-means on UMAP-150D coordinates, and calls the difference a "13-17x silhouette
improvement". UMAP optimizes local neighbourhood structure, so a silhouette computed on
UMAP coordinates is inflated by construction and is not commensurable with one computed in
the original space.

This script settles the question on a random subsample: it fits UMAP and k-means on the
same sample, then scores EVERY labeling in the ORIGINAL embedding space, which is the
comparison the reviewers ask for. It also reports each labeling's silhouette in the space
it was fitted in, so the size of the inflation is visible.

Subsample, not the full corpus: UMAP over 200,525 x 1536 is hours of CPU, and the
methodological question does not need corpus scale. The sample size and seed are recorded
below and the script is deterministic.

Class B (no LLM, no network). Run from graph_analysis/:
    python -u experiment_review_silhouette_original_space.py [--n 30000]

Output: graph_analysis/phase2_results/experiment_review_silhouette_report.json
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
SRC = STEP1 / "graph_node_attributes.pkl"
OUT = ROOT / "phase2_results/experiment_review_silhouette_report.json"

SEED = 42  # the deployed clustering stage uses random_state=42; mirror it
SIL_SAMPLE = 5000  # nodes used for the silhouette estimate (pairwise distances)
AGGLO_N = 10000  # agglomerative linkage is O(n^2) in memory; run it on a subset
# UMAP configuration copied from the deployed clustering stage (phase2_clustering.py):
UMAP_KW = dict(
    n_components=150, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=SEED
)


def parse_embedding(s):
    if s is None:
        return None
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.float32)
    # The Step-1 checkpoint stores embeddings as FalkorDB vector literals: "<a, b, ...>"
    s = s.strip().lstrip("<[").rstrip(">]")
    return np.fromstring(s, sep=",", dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="subsample size")
    a = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(
            f"FATAL: {SRC} not found; no fallback embedding source exists."
        )

    t0 = time.time()
    na = pickle.load(open(SRC, "rb"))
    ids = sorted(k for k, v in na.items() if v.get("embedding") is not None)
    rng = random.Random(SEED)
    sample = sorted(rng.sample(ids, min(a.n, len(ids))))
    print(
        f"{len(ids):,} nodes carry an embedding; sampling {len(sample):,}", flush=True
    )

    X = np.vstack([parse_embedding(na[i]["embedding"]) for i in sample])
    cats = [
        (na[i].get("concept_category") or na[i].get("type") or "?").lower()
        for i in sample
    ]
    del na
    print(f"embedding matrix {X.shape} in {time.time() - t0:.0f}s", flush=True)
    if X.shape[1] != 1536:
        print(f"NOTE: embedding dimension is {X.shape[1]}, not 1536", flush=True)

    import umap  # imported late: heavy, and only this script needs it

    t1 = time.time()
    U = umap.UMAP(**UMAP_KW).fit_transform(X)
    print(f"UMAP {U.shape} in {time.time() - t1:.0f}s", flush=True)

    def km(mat, k):
        return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(mat)

    labelings = {}
    for k in (5, 8, 40):
        labelings[f"kmeans_k{k}_fitted_in_1536D"] = (km(X, k), "1536D")
        print(f"  fitted k-means k={k} in original space", flush=True)
    for k in (8, 40):
        labelings[f"kmeans_k{k}_fitted_on_UMAP150D"] = (km(U, k), "UMAP150D")
        print(f"  fitted k-means k={k} on UMAP coordinates", flush=True)

    rows = {}
    for name, (lab, fitted_in) in labelings.items():
        row = {
            "fitted_in": fitted_in,
            "n_clusters": int(len(set(lab))),
            "silhouette_in_ORIGINAL_1536D_euclidean": round(
                float(
                    silhouette_score(X, lab, sample_size=SIL_SAMPLE, random_state=SEED)
                ),
                4,
            ),
            "silhouette_in_ORIGINAL_1536D_cosine": round(
                float(
                    silhouette_score(
                        X,
                        lab,
                        metric="cosine",
                        sample_size=SIL_SAMPLE,
                        random_state=SEED,
                    )
                ),
                4,
            ),
            "calinski_harabasz_in_ORIGINAL_1536D": round(
                float(calinski_harabasz_score(X, lab)), 1
            ),
            "davies_bouldin_in_ORIGINAL_1536D": round(
                float(davies_bouldin_score(X, lab)), 3
            ),
        }
        if fitted_in == "UMAP150D":
            row["silhouette_in_UMAP150D_euclidean"] = round(
                float(
                    silhouette_score(U, lab, sample_size=SIL_SAMPLE, random_state=SEED)
                ),
                4,
            )
        rows[name] = row
        print(f"  scored {name}", flush=True)

    # ---- the algorithm the deployed clustering stage actually used -------------------
    # phase2_clustering.py clusters the UMAP-150D coordinates with AgglomerativeClustering
    # (cosine, average linkage) at k=40, not k-means. Linkage is O(n^2) in memory, so this
    # block runs on a subset of the same sample and is scored on that subset.
    sub = np.sort(
        np.random.default_rng(SEED).choice(len(sample), size=AGGLO_N, replace=False)
    )
    Xs, Us = X[sub], U[sub]
    agglo_rows = {}
    for space, mat, tag in (
        (("UMAP150D"), Us, "on_UMAP150D"),
        ("1536D", Xs, "in_1536D"),
    ):
        lab = AgglomerativeClustering(
            n_clusters=40, metric="cosine", linkage="average"
        ).fit_predict(mat)
        agglo_rows[f"agglomerative_k40_fitted_{tag}"] = {
            "fitted_in": space,
            "n_nonempty_clusters": int(len(set(lab))),
            "largest_cluster_share_pct": round(
                100 * max(np.bincount(lab)) / len(lab), 1
            ),
            "silhouette_in_ORIGINAL_1536D_euclidean": round(
                float(
                    silhouette_score(Xs, lab, sample_size=SIL_SAMPLE, random_state=SEED)
                ),
                4,
            ),
            "silhouette_in_ORIGINAL_1536D_cosine": round(
                float(
                    silhouette_score(
                        Xs,
                        lab,
                        metric="cosine",
                        sample_size=SIL_SAMPLE,
                        random_state=SEED,
                    )
                ),
                4,
            ),
        }
        if space == "UMAP150D":
            agglo_rows[f"agglomerative_k40_fitted_{tag}"][
                "silhouette_in_UMAP150D_euclidean"
            ] = round(
                float(
                    silhouette_score(Us, lab, sample_size=SIL_SAMPLE, random_state=SEED)
                ),
                4,
            )
        print(f"  scored agglomerative k=40 fitted {tag}", flush=True)

    # what the UMAP labels' apparent gain looks like when both are scored in one space
    u40 = rows["kmeans_k40_fitted_on_UMAP150D"]
    d40 = rows["kmeans_k40_fitted_in_1536D"]
    d8 = rows["kmeans_k8_fitted_in_1536D"]

    out = {
        "experiment": "silhouette of UMAP-fitted vs original-space clusterings, scored in one space (W-6)",
        "subsample": {"n": len(sample), "seed": SEED, "silhouette_sample": SIL_SAMPLE},
        "embedding_dim": int(X.shape[1]),
        "umap": UMAP_KW,
        "labelings": rows,
        "deployed_algorithm_agglomerative": {
            "n_subset": AGGLO_N,
            "config": "AgglomerativeClustering(n_clusters=40, metric=cosine, linkage=average)",
            "rows": agglo_rows,
        },
        "headline": {
            "umap_k40_silhouette_in_its_own_space": u40.get(
                "silhouette_in_UMAP150D_euclidean"
            ),
            "umap_k40_silhouette_in_original_space": u40[
                "silhouette_in_ORIGINAL_1536D_euclidean"
            ],
            "direct_k40_silhouette_in_original_space": d40[
                "silhouette_in_ORIGINAL_1536D_euclidean"
            ],
            "direct_k8_silhouette_in_original_space": d8[
                "silhouette_in_ORIGINAL_1536D_euclidean"
            ],
            "READING": (
                "A silhouette computed on UMAP coordinates is not comparable with one "
                "computed in the original space. Compare the two ORIGINAL-space rows to "
                "each other: that is the question of whether the UMAP step bought better "
                "clusters or a better-looking score."
            ),
        },
        "sample_category_mix": {
            c: cats.count(c) for c in sorted(set(cats), key=lambda x: -cats.count(x))
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["headline"], indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
