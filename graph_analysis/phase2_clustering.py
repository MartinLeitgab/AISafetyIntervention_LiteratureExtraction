"""
Phase 2: Pathway Node Clustering Analysis
Cluster nodes across edge configs × modes × node types to identify distinct mechanisms
"""

import redis
import numpy as np
import json
import os
import gc
import time
import threading
from collections import defaultdict, Counter
from multiprocessing import Pool
from contextlib import contextmanager
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
import community as community_louvain
import networkx as nx
from umap import UMAP

try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None

# ============================================================================
# CONFIGURATION
# ============================================================================

client = redis.Redis(host="localhost", port=6379, decode_responses=True)
graph = "AISafetyIntervention"

EDGE_CONFIGS = ["EDGE", 0.80, 0.85, 0.90, 0.95]
MODES = ["unconstrained", "single_risk", "monotonic", "both"]
NODE_TYPES = [
    "risk",
    "intervention",
    "all_concepts",
    "problem_analysis",
    "theoretical_insight",
    "design_rationale",
    "implementation_mechanism",
    "validation_evidence",
]

N_CLUSTERS_TARGET = (40, 60)
MIN_CLUSTER_SIZE = 5
K_VALUES = [20, 30, 40, 50, 60]  # Test multiple cluster counts

# ============================================================================
# DATABASE QUERIES WITH RETRY
# ============================================================================


def query_with_retry(q, timeout=300000, max_retries=3):
    """Execute query with exponential backoff"""
    for attempt in range(max_retries):
        try:
            result = client.execute_command(
                "GRAPH.QUERY", graph, q, "--timeout", str(timeout)
            )
            return result[1] if len(result) > 1 else []
        except Exception as e:
            if "timed out" in str(e).lower() and attempt < max_retries - 1:
                wait = 2**attempt
                time.sleep(wait)
                continue
            raise


def fetch_node_metadata(node_ids, batch_size=100):
    """Fetch name, category for nodes in small batches"""
    metadata = {}
    node_list = list(node_ids)

    for i in range(0, len(node_list), batch_size):
        batch = node_list[i : i + batch_size]
        id_str = ",".join(str(nid) for nid in batch)

        q = f"""
        MATCH (n)
        WHERE id(n) IN [{id_str}]
        RETURN id(n), n.name, n.concept_category
        """

        for row in query_with_retry(q):
            node_id = int(row[0])
            metadata[node_id] = {
                "name": row[1],
                "category": row[2] if row[2] else "unknown",
            }

    return metadata


def fetch_embeddings_batch(node_ids, metadata_dict, batch_size=100):
    """Fetch embeddings in small batches with retry"""
    embeddings = {}
    node_list = list(node_ids)
    missing_by_category = defaultdict(int)

    for i in range(0, len(node_list), batch_size):
        batch = node_list[i : i + batch_size]
        id_str = ",".join(str(nid) for nid in batch)

        q = f"""
        MATCH (n)
        WHERE id(n) IN [{id_str}]
        RETURN id(n), n.embedding
        """

        for row in query_with_retry(q):
            node_id = int(row[0])
            emb_str = row[1]
            if emb_str:
                try:
                    emb_str_clean = emb_str.strip("<>")
                    emb_array = np.fromstring(emb_str_clean, sep=",", dtype=np.float32)
                    if len(emb_array) == 1536:
                        embeddings[node_id] = emb_array
                    else:
                        cat = metadata_dict.get(node_id, {}).get("category", "unknown")
                        missing_by_category[cat] += 1
                except:
                    cat = metadata_dict.get(node_id, {}).get("category", "unknown")
                    missing_by_category[cat] += 1
            else:
                cat = metadata_dict.get(node_id, {}).get("category", "unknown")
                missing_by_category[cat] += 1

    return embeddings, missing_by_category


# ============================================================================
# PATH LOADING
# ============================================================================


def load_nodes_from_paths(path_file, node_type):
    """Load unique nodes from path file filtered by type"""
    nodes = set()
    path_count = 0

    if not os.path.exists(path_file):
        return nodes, 0

    with open(path_file, "r") as f:
        for line in f:
            data = json.loads(line)
            path_nodes = data["path"]
            categories = data["categories"]

            for node_id, cat in zip(path_nodes, categories):
                if node_type == "risk" and cat == "risk":
                    nodes.add(node_id)
                elif node_type == "intervention" and cat == "intervention":
                    nodes.add(node_id)
                elif node_type == "all_concepts" and cat not in [
                    "risk",
                    "intervention",
                ]:
                    nodes.add(node_id)
                elif node_type == cat.replace(" ", "_"):
                    nodes.add(node_id)

            path_count += 1

    return nodes, path_count


def identify_edge_only_nodes(edge_path_file):
    """Extract nodes appearing in EDGE-only paths"""
    nodes = set()

    if not os.path.exists(edge_path_file):
        return nodes

    with open(edge_path_file, "r") as f:
        for line in f:
            data = json.loads(line)
            nodes.update(data["path"])

    return nodes


# ============================================================================
# CLUSTERING ALGORITHMS
# ============================================================================


def cluster_hdbscan(embeddings_matrix, min_cluster_size=MIN_CLUSTER_SIZE):
    """HDBSCAN clustering"""
    if HDBSCAN is None:
        return None, 0

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings_matrix)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters


def cluster_agglomerative(embeddings_matrix, n_clusters=50):
    """Agglomerative clustering with cosine distance"""
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    labels = clusterer.fit_predict(embeddings_matrix)

    return labels, n_clusters


def cluster_louvain(embeddings_matrix, node_ids):
    """Louvain community detection on k-NN graph"""
    k = 10
    n = len(embeddings_matrix)
    G = nx.Graph()

    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dist = cosine(embeddings_matrix[i], embeddings_matrix[j])
                dists.append((j, dist))

        dists.sort(key=lambda x: x[1])
        for j, dist in dists[:k]:
            G.add_edge(node_ids[i], node_ids[j], weight=1 - dist)

    partition = community_louvain.best_partition(G)
    labels = np.array([partition[nid] for nid in node_ids])
    n_clusters = len(set(labels))

    return labels, n_clusters


# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================


def compute_exemplar(cluster_members, embeddings_dict):
    """Find most central node"""
    member_list = list(cluster_members)

    if len(member_list) == 1:
        return member_list[0]

    min_dist = float("inf")
    exemplar = member_list[0]

    for node_i in member_list:
        emb_i = embeddings_dict[node_i]
        avg_dist = 0

        for node_j in member_list:
            if node_i != node_j:
                emb_j = embeddings_dict[node_j]
                avg_dist += cosine(emb_i, emb_j)

        avg_dist /= len(member_list) - 1

        if avg_dist < min_dist:
            min_dist = avg_dist
            exemplar = node_i

    return exemplar


def extract_top_terms(cluster_members, metadata_dict, top_n=5):
    """Extract most frequent terms from member names"""
    all_terms = []

    for node_id in cluster_members:
        if node_id in metadata_dict:
            name = metadata_dict[node_id]["name"]
            terms = name.lower().replace("-", " ").split()
            stopwords = {
                "the",
                "a",
                "an",
                "of",
                "in",
                "to",
                "for",
                "with",
                "from",
                "by",
                "as",
                "at",
                "on",
                "is",
                "are",
                "and",
                "or",
            }
            terms = [t for t in terms if t not in stopwords and len(t) > 2]
            all_terms.extend(terms)

    term_counts = Counter(all_terms)
    return [term for term, _ in term_counts.most_common(top_n)]


def compute_edge_validation_rate(cluster_members, edge_only_nodes):
    """Fraction of cluster with EDGE-only evidence"""
    if not edge_only_nodes:
        return 0.0

    edge_count = sum(1 for nid in cluster_members if nid in edge_only_nodes)
    return edge_count / len(cluster_members) if cluster_members else 0.0


def analyze_clusters(labels, node_ids, embeddings_dict, metadata_dict, edge_only_nodes):
    """Generate cluster metadata"""
    cluster_data = {}

    clusters = defaultdict(set)
    for node_id, label in zip(node_ids, labels):
        if label >= 0:
            clusters[label].add(node_id)

    for cluster_id, members in clusters.items():
        exemplar = compute_exemplar(members, embeddings_dict)
        top_terms = extract_top_terms(members, metadata_dict)
        edge_rate = compute_edge_validation_rate(members, edge_only_nodes)

        cluster_data[cluster_id] = {
            "size": len(members),
            "exemplar": exemplar,
            "exemplar_name": metadata_dict.get(exemplar, {}).get("name", "unknown"),
            "top_terms": top_terms,
            "edge_validation_rate": edge_rate,
            "members": list(members),
        }

    return cluster_data


def compute_validation_metrics(
    labels, embeddings_matrix, cluster_data, edge_only_nodes
):
    """Compute validation metrics"""
    metrics = {}

    valid_mask = labels >= 0
    if valid_mask.sum() > 1:
        try:
            metrics["silhouette"] = silhouette_score(
                embeddings_matrix[valid_mask], labels[valid_mask], metric="cosine"
            )
        except:
            metrics["silhouette"] = -1
    else:
        metrics["silhouette"] = -1

    if edge_only_nodes:
        total_nodes = sum(c["size"] for c in cluster_data.values())
        total_edge = sum(
            c["edge_validation_rate"] * c["size"] for c in cluster_data.values()
        )
        metrics["edge_validation_overall"] = (
            total_edge / total_nodes if total_nodes > 0 else 0
        )
    else:
        metrics["edge_validation_overall"] = 0

    sizes = [c["size"] for c in cluster_data.values()]
    if sizes:
        metrics["cluster_size_mean"] = np.mean(sizes)
        metrics["cluster_size_median"] = np.median(sizes)
        metrics["cluster_size_min"] = np.min(sizes)
        metrics["cluster_size_max"] = np.max(sizes)

    return metrics


# ============================================================================
# MAIN CLUSTERING FUNCTION
# ============================================================================


def cluster_config(config_tuple):
    """Process one configuration with step-level buffering"""
    edge_config, mode, node_type, config_idx, total_configs = config_tuple
    worker_id = os.getpid() % 10

    if edge_config == "EDGE":
        config_label = f"EDGE_{mode}_{node_type}"
        path_file = f"phase1_rawpathsfiles/paths_{mode}_edge_only.jsonl"
    else:
        config_label = f"SIM{edge_config}_{mode}_{node_type}"
        path_file = f"phase1_rawpathsfiles/paths_{mode}_sim{edge_config}.jsonl"

    output_file = f"clusters_{config_label}.json"

    if os.path.exists(output_file):
        print(
            f"W{worker_id} [{config_idx}/{total_configs}] ⏭ Skipping {config_label} (exists)",
            flush=True,
        )
        return config_label, None

    step_buffer = []

    @contextmanager
    def step(num, name):
        step_buffer.clear()
        step_buffer.append(f"\nW{worker_id} [{config_idx}/{total_configs}] {name}")
        log_fn = lambda msg: step_buffer.append(f"W{worker_id}     {msg}")
        try:
            yield log_fn
        except Exception as e:
            step_buffer.append(f"W{worker_id}     ❌ ERROR: {e}")
            print("\n".join(step_buffer), flush=True)
            raise
        else:
            print("\n".join(step_buffer), flush=True)

    start_time = time.time()

    # Step 1: Load nodes
    with step(1, f"[1/7] Loading nodes - {config_label}") as log:
        edge_path_file = f"phase1_rawpathsfiles/paths_{mode}_edge_only.jsonl"
        edge_only_nodes = identify_edge_only_nodes(edge_path_file)
        log(f"EDGE validation: {len(edge_only_nodes):,} nodes")

        nodes, path_count = load_nodes_from_paths(path_file, node_type)

        if len(nodes) < MIN_CLUSTER_SIZE:
            log(f"⚠ Too few nodes ({len(nodes)}), skipping")
            return config_label, {"status": "insufficient_data", "n_nodes": len(nodes)}

        log(f"✓ {len(nodes):,} nodes, {path_count:,} paths")

    # Step 2: Fetch metadata
    with step(2, "[2/7] Fetching metadata") as log:
        metadata_dict = fetch_node_metadata(nodes)
        log(f"✓ {len(metadata_dict):,} nodes")

    # Step 3: Fetch embeddings
    with step(3, "[3/7] Fetching embeddings") as log:
        embeddings_dict, missing_by_cat = fetch_embeddings_batch(nodes, metadata_dict)

        total_missing = sum(missing_by_cat.values())
        log(f"✓ {len(embeddings_dict):,} embeddings ({total_missing:,} missing)")

        if total_missing > 0:
            log(f"Missing: {dict(missing_by_cat)}")

        if len(embeddings_dict) < MIN_CLUSTER_SIZE:
            log(f"⚠ Too few embeddings ({len(embeddings_dict)})")
            return config_label, {
                "status": "insufficient_embeddings",
                "n_embeddings": len(embeddings_dict),
                "missing_by_category": dict(missing_by_cat),
            }

    # Step 4: Prepare matrix
    with step(4, "[4/7] Preparing matrix") as log:
        node_ids = list(embeddings_dict.keys())
        embeddings_matrix = np.array([embeddings_dict[nid] for nid in node_ids])
        log(f"✓ {embeddings_matrix.shape}")

    # Step 4.5: UMAP dimensionality reduction
    with step(4.5, "[4.5/7] UMAP reduction 1536D→150D") as log:
        reducer = UMAP(n_components=150, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_matrix = reducer.fit_transform(embeddings_matrix)
        log(f"✓ Reduced to {embeddings_matrix.shape}")

    # Step 5: Clustering
    with step(5, "[5/7] Clustering") as log:
        results = {}

        if HDBSCAN is not None:
            labels_hdbscan, n_hdbscan = cluster_hdbscan(embeddings_matrix)
            if labels_hdbscan is not None:
                log(f"HDBSCAN: {n_hdbscan} clusters")
                results["hdbscan"] = {"labels": labels_hdbscan, "n_clusters": n_hdbscan}

        labels_agg, _ = cluster_agglomerative(embeddings_matrix, n_clusters=40)
        log("Agglomerative k=40: 40 clusters")
        results["agglomerative"] = {"labels": labels_agg, "n_clusters": 40}

        labels_louvain, n_louvain = cluster_louvain(embeddings_matrix, node_ids)
        log(f"Louvain: {n_louvain} clusters")
        results["louvain"] = {"labels": labels_louvain, "n_clusters": n_louvain}

    # Step 6: Analysis
    with step(6, "[6/7] Analyzing") as log:
        for algo_name in results:
            labels = results[algo_name]["labels"]

            cluster_data = analyze_clusters(
                labels, node_ids, embeddings_dict, metadata_dict, edge_only_nodes
            )

            metrics = compute_validation_metrics(
                labels, embeddings_matrix, cluster_data, edge_only_nodes
            )

            results[algo_name].update(
                {
                    "cluster_data": cluster_data,
                    "metrics": {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metrics.items()
                    },
                    "assignments": {
                        int(nid): int(label) for nid, label in zip(node_ids, labels)
                    },
                }
            )
            del results[algo_name]["labels"]

            log(
                f"{algo_name}: Sil={metrics['silhouette']:.3f}, EDGE={metrics['edge_validation_overall']:.3f}"
            )

    # Step 7: Save
    with step(7, "[7/7] Saving") as log:

        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        output = {
            "config": config_label,
            "edge_config": edge_config,
            "mode": mode,
            "node_type": node_type,
            "n_nodes": len(nodes),
            "n_paths": path_count,
            "n_edge_only": len(edge_only_nodes),
            "n_embeddings": len(embeddings_dict),
            "missing_by_category": dict(missing_by_cat),
            "results": convert_numpy(results),
            "runtime_seconds": time.time() - start_time,
        }

        output_json = json.dumps(output, indent=2)

        with open(output_file, "w") as f:
            f.write(output_json)

        log(f"✓ {output_file}")
        log(f"Runtime: {time.time() - start_time:.1f}s")

    del embeddings_dict, embeddings_matrix, metadata_dict
    gc.collect()

    return config_label, output


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================


def run_all_clustering():
    """Run all 160 configurations"""

    print(f"\n{'#' * 80}")
    print("PHASE 2: PATHWAY NODE CLUSTERING")
    print(f"{'#' * 80}")
    total_configs = len(EDGE_CONFIGS) * len(MODES) * len(NODE_TYPES)
    print(
        f"Total: {len(EDGE_CONFIGS)} × {len(MODES)} × {len(NODE_TYPES)} = {total_configs}"
    )

    all_configs = []
    # all_configs = [(0.85, 'unconstrained', 'risk', 1, 1)]

    idx = 1
    for edge_config in EDGE_CONFIGS:
        for mode in MODES:
            for node_type in NODE_TYPES:
                all_configs.append((edge_config, mode, node_type, idx, total_configs))
                idx += 1

    print("\n4 workers with retry + small batches...")

    n_workers = 4  # 4 prod, 1 for deubg/test

    def print_progress():
        while True:
            time.sleep(30)
            current = len(
                [
                    f
                    for f in os.listdir(".")
                    if f.startswith("clusters_") and f.endswith(".json")
                ]
            )
            elapsed = time.time() - start_time
            eta = (
                (elapsed / current * (total_configs - current)) / 60
                if current > 0
                else 0
            )
            print(f"\n{'=' * 80}")
            print(
                f"PROGRESS: {current}/{total_configs} ({100 * current / total_configs:.1f}%) | {elapsed / 60:.1f}min | ETA {eta:.1f}min"
            )
            print(f"{'=' * 80}\n", flush=True)

    monitor = threading.Thread(target=print_progress, daemon=True)
    monitor.start()

    start_time = time.time()

    with Pool(processes=n_workers) as pool:
        results = pool.map(cluster_config, all_configs)

    total_time = time.time() - start_time

    print(f"\n{'#' * 80}")
    print(f"COMPLETE: {total_time / 3600:.2f} hours")
    print(f"{'#' * 80}")

    success = sum(1 for _, r in results if r and "results" in r)
    print(f"Success: {success}/{len(results)}\n")

    return results


if __name__ == "__main__":
    run_all_clustering()
