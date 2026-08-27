#!/usr/bin/env python3
"""
Stage 2.1 Alternative: Semantic clustering с UMAP dimension reduction

UMAP снижает размерность embeddings (1536 -> 50-100),
затем K-means на сниженном пространстве.
Часто даёт лучший silhouette score.
"""

import pickle
import logging
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap


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
    logger.info(f"Загрузка графа из {path}...")
    with open(path, 'rb') as f:
        G = pickle.load(f)
    logger.info(f"Загружено: {G.number_of_nodes():,} нод, {G.number_of_edges():,} рёбер")
    return G


def extract_embeddings(G):
    logger = logging.getLogger(__name__)
    logger.info("Извлечение embeddings из нод...")
    
    node_ids = []
    embeddings = []
    
    for node_id, data in tqdm(G.nodes(data=True), desc="Извлечение embeddings"):
        if 'embedding' in data and data['embedding'] is not None:
            node_ids.append(node_id)
            embeddings.append(data['embedding'])
    
    embeddings = np.array(embeddings)
    logger.info(f"Извлечено {len(embeddings):,} embeddings")
    
    skipped = G.number_of_nodes() - len(embeddings)
    if skipped > 0:
        logger.warning(f"⚠️  {skipped} нод без embeddings (пропущены)")
    
    return node_ids, embeddings


def reduce_dimensions_umap(embeddings, n_components=50, n_neighbors=15, min_dist=0.0, 
                           sample_size=None, random_state=42):
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("UMAP: снижение размерности")
    logger.info("="*60)
    logger.info(f"Исходная размерность: {embeddings.shape[1]}")
    logger.info(f"Целевая размерность: {n_components}")
    logger.info(f"n_neighbors: {n_neighbors}")
    logger.info(f"min_dist: {min_dist}")
    
    # Для ускорения можем использовать выборку
    if sample_size and sample_size < len(embeddings):
        logger.info(f"Используем выборку {sample_size:,} из {len(embeddings):,} для обучения UMAP")
        sample_indices = np.random.RandomState(random_state).choice(
            len(embeddings), size=sample_size, replace=False
        )
        embeddings_sample = embeddings[sample_indices]
    else:
        embeddings_sample = embeddings
        logger.info(f"Используем все {len(embeddings):,} точек")
    
    # Обучение UMAP
    logger.info("Обучение UMAP...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=random_state,
        verbose=True
    )
    
    if sample_size and sample_size < len(embeddings):
        # Обучаем на выборке
        reducer.fit(embeddings_sample)
        logger.info("Трансформация всех embeddings...")
        embeddings_reduced = reducer.transform(embeddings)
    else:
        # Обучаем и трансформируем все сразу
        embeddings_reduced = reducer.fit_transform(embeddings)
    
    logger.info(f"✅ UMAP завершён: {embeddings.shape} → {embeddings_reduced.shape}")
    
    return embeddings_reduced, reducer


def load_umap_reducer(reducer_path):
    """Загружает обученную UMAP модель из файла"""
    logger = logging.getLogger(__name__)
    logger.info(f"Загрузка UMAP модели из {reducer_path}...")
    
    with open(reducer_path, 'rb') as f:
        reducer = pickle.load(f)
    
    logger.info(f"✅ UMAP модель загружена")
    return reducer


def find_optimal_k_elbow(embeddings, k_values, sample_size=10000):
    logger = logging.getLogger(__name__)
    
    # Используем выборку для ускорения
    if len(embeddings) > sample_size:
        logger.info(f"Используем выборку {sample_size:,} из {len(embeddings):,} нод для ускорения")
        sample_indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings_sample = embeddings[sample_indices]
    else:
        embeddings_sample = embeddings
    
    results = []
    
    for k in k_values:
        logger.info(f"\n--- Тестирование k={k} ---")
        
        kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(embeddings_sample)
        
        inertia = kmeans.inertia_
        silhouette = silhouette_score(embeddings_sample, labels, metric='euclidean', sample_size=min(5000, len(embeddings_sample)))
        
        logger.info(f"  Inertia: {inertia:,.2f}")
        logger.info(f"  Silhouette: {silhouette:.4f}")
        
        results.append({
            'k': k,
            'inertia': float(inertia),
            'silhouette': float(silhouette)
        })
    
    return results


def plot_elbow_curve(elbow_results, output_path):
    logger = logging.getLogger(__name__)
    
    k_values = [r['k'] for r in elbow_results]
    inertias = [r['inertia'] for r in elbow_results]
    silhouettes = [r['silhouette'] for r in elbow_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Стиль для научной публикации
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    # График Inertia
    ax1.plot(k_values, inertias, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (within-cluster sum of squares)', fontsize=12, fontweight='bold')
    ax1.set_title('Elbow Method: Inertia', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(k_values)
    
    # График Silhouette
    ax2.plot(k_values, silhouettes, 's-', color='#A23B72', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of clusters (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Silhouette Score by k', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(k_values)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Отмечаем лучший k
    best_idx = silhouettes.index(max(silhouettes))
    ax2.plot(k_values[best_idx], silhouettes[best_idx], 'r*', markersize=15, 
             label=f'Best k={k_values[best_idx]}')
    ax2.legend()
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    logger.info(f"✅ Elbow plot сохранён: {output_path} (300 DPI)")


def perform_clustering(embeddings, k):
    logger = logging.getLogger(__name__)
    logger.info(f"K-means кластеризация (k={k})...")
    
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    
    logger.info(f"✅ Кластеризация завершена")
    return labels, centroids


def compute_cluster_info(node_ids, embeddings, labels, centroids):
    logger = logging.getLogger(__name__)
    logger.info("Вычисление метаданных кластеров...")
    
    clusters_info = defaultdict(lambda: {'nodes': [], 'embeddings': []})
    
    for node_id, embedding, label in zip(node_ids, embeddings, labels):
        clusters_info[int(label)]['nodes'].append(node_id)
        clusters_info[int(label)]['embeddings'].append(embedding)
    
    # Для каждого кластера найти 5 ближайших к центроиду нод
    result = {}
    for cluster_id, info in clusters_info.items():
        centroid = centroids[cluster_id]
        cluster_embeddings = np.array(info['embeddings'])
        
        # Евклидово расстояние до центроида
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_indices = np.argsort(distances)[:5]
        
        representatives = [info['nodes'][i] for i in closest_indices]
        
        result[cluster_id] = {
            'size': len(info['nodes']),
            'representatives': representatives,
            'centroid': centroid.tolist()
        }
    
    logger.info(f"✅ Метаданные для {len(result)} кластеров")
    
    # Статистика
    sizes = [info['size'] for info in result.values()]
    logger.info(f"\nРаспределение размеров кластеров:")
    logger.info(f"  Min: {min(sizes)}")
    logger.info(f"  Max: {max(sizes)}")
    logger.info(f"  Средний: {np.mean(sizes):.1f}")
    logger.info(f"  Медиана: {np.median(sizes):.1f}")
    
    return result


def compute_cluster_distances(centroids, threshold=0.3):
    logger = logging.getLogger(__name__)
    logger.info(f"Вычисление расстояний между кластерами (threshold={threshold})...")
    
    # Косинусное расстояние между всеми парами центроидов
    distances = cosine_distances(centroids)
    
    # Для каждого кластера находим ближайшие (distance < threshold)
    nearest_clusters = {}
    for i in range(len(centroids)):
        cluster_distances = []
        for j in range(len(centroids)):
            if i != j and distances[i, j] < threshold:
                cluster_distances.append({
                    'cluster_id': int(j),
                    'distance': float(distances[i, j])
                })
        
        # Сортируем по расстоянию
        cluster_distances.sort(key=lambda x: x['distance'])
        nearest_clusters[i] = cluster_distances
    
    logger.info(f"✅ Вычислено {len(nearest_clusters)} пар близких кластеров")
    return nearest_clusters


def add_clusters_to_graph(G, node_ids, labels):
    logger = logging.getLogger(__name__)
    logger.info("Добавление атрибута semantic_cluster в граф...")
    
    for node_id, label in zip(node_ids, labels):
        G.nodes[node_id]['semantic_cluster'] = int(label)
    
    logger.info(f"✅ Атрибут добавлен для {len(node_ids):,} нод")


def save_clusters_json(clusters_info, nearest_clusters, G, output_path, metadata):
    logger = logging.getLogger(__name__)
    logger.info(f"Сохранение JSON для разметки: {output_path}")
    
    # Формируем итоговую структуру
    clusters = []
    for cluster_id, info in sorted(clusters_info.items()):
        # Получаем имена представителей
        representative_nodes = []
        for node_id in info['representatives']:
            node_data = G.nodes[node_id]
            representative_nodes.append({
                'node_id': node_id,
                'name': node_data.get('name', 'N/A')
            })
        
        clusters.append({
            'cluster_id': cluster_id,
            'size': info['size'],
            'centroid_embedding': info['centroid'],
            'representative_nodes': representative_nodes,
            'nearest_clusters': nearest_clusters.get(cluster_id, [])
        })
    
    output_data = {
        'metadata': metadata,
        'clusters': clusters
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ JSON сохранён: {len(clusters)} кластеров")
    
    # Показываем несколько примеров
    logger.info("\nПримеры кластеров:")
    for cluster in sorted(clusters, key=lambda x: x['size'], reverse=True)[:5]:
        logger.info(f"  Кластер {cluster['cluster_id']}: {cluster['size']} нод")
        logger.info(f"    Представители: {', '.join([r['name'][:50] for r in cluster['representative_nodes'][:3]])}")


def save_umap_reducer(reducer, output_dir, n_components, n_neighbors, original_dims, timestamp):
    """Сохраняет UMAP reducer и метаданные"""
    logger = logging.getLogger(__name__)
    
    # Сохраняем в results/umap_models/
    models_dir = output_dir / 'umap_models'
    models_dir.mkdir(exist_ok=True, parents=True)
    
    reducer_path = models_dir / f'stage2.1_umap_reducer_{original_dims}d_to_{n_components}d.pkl'
    with open(reducer_path, 'wb') as f:
        pickle.dump(reducer, f)
    
    logger.info(f"✅ UMAP reducer сохранён: {reducer_path}")
    
    metadata = {
        'stage': '2.1',
        'original_dims': original_dims,
        'target_dims': n_components,
        'n_neighbors': n_neighbors,
        'min_dist': reducer.min_dist,
        'metric': reducer.metric,
        'random_state': reducer.random_state,
        'timestamp': timestamp,
        'reducer_path': str(reducer_path)
    }
    
    metadata_path = models_dir / 'stage2.1_umap_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ UMAP metadata сохранён: {metadata_path}")
    
    return reducer_path


def add_umap_embeddings_to_graph(G, node_ids, embeddings_reduced, n_components):
    """Добавляет сжатые UMAP embeddings в граф"""
    logger = logging.getLogger(__name__)
    logger.info(f"\nДобавление UMAP embeddings ({n_components}D) в граф...")
    
    attr_name = f'embedding_umap_{n_components}d'
    
    for node_id, embedding in tqdm(zip(node_ids, embeddings_reduced), 
                                    total=len(node_ids),
                                    desc=f"Добавление {attr_name}"):
        G.nodes[node_id][attr_name] = embedding.tolist()
    
    logger.info(f"✅ Атрибут '{attr_name}' добавлен для {len(node_ids):,} нод")


def main():
    parser = argparse.ArgumentParser(description='Stage 2.1 Alternative: UMAP + K-means clustering')
    parser.add_argument('--graph_path', required=True, help='Путь к графу после Stage 1')
    parser.add_argument('--output', required=True, help='Путь для сохранения графа с кластерами')
    parser.add_argument('--umap_dims', type=int, default=50, help='UMAP target dimensions (default: 50)')
    parser.add_argument('--umap_neighbors', type=int, default=15, help='UMAP n_neighbors (default: 15)')
    parser.add_argument('--umap_sample', type=int, default=None, help='Sample size for UMAP training (default: all)')
    parser.add_argument('--load_umap', type=str, help='Путь к сохранённой UMAP модели (для переиспользования)')
    parser.add_argument('--k_values', type=str, help='Список k для elbow method (через запятую)')
    parser.add_argument('--chosen_k', type=int, help='Конкретное k для кластеризации')
    parser.add_argument('--output_suffix', type=str, default='', help='Постфикс для имен выходных файлов (например, "_refined")')
    parser.add_argument('--log', type=str, help='Путь к лог файлу')
    
    args = parser.parse_args()
    
    logger = setup_logging(args.log)
    
    logger.info("="*60)
    logger.info("STAGE 2.1 ALTERNATIVE: UMAP + Semantic clustering")
    logger.info("="*60)
    logger.info(f"Входной граф: {args.graph_path}")
    logger.info(f"Выходной граф: {args.output}")
    logger.info(f"UMAP dimensions: {args.umap_dims}")
    if args.log:
        logger.info(f"Лог файл: {args.log}")
    
    # 1. Загрузка графа
    G = load_graph(args.graph_path)
    
    # 2. Извлечение embeddings
    node_ids, embeddings = extract_embeddings(G)
    
    # 3. UMAP dimension reduction
    logger.info("\n" + "="*60)
    logger.info("UMAP: снижение размерности")
    logger.info("="*60)
    
    if args.load_umap:
        # Загружаем существующую модель
        reducer = load_umap_reducer(args.load_umap)
        logger.info(f"Исходная размерность: {embeddings.shape[1]}")
        logger.info(f"Трансформация embeddings с помощью загруженной модели...")
        embeddings_reduced = reducer.transform(embeddings)
        logger.info(f"✅ Трансформация завершена: {embeddings.shape} → {embeddings_reduced.shape}")
        
        # Добавляем UMAP embeddings в граф
        add_umap_embeddings_to_graph(G, node_ids, embeddings_reduced, embeddings_reduced.shape[1])
    else:
        # Обучаем новую модель
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        logger.info(f"Исходная размерность: {embeddings.shape[1]}")
        logger.info(f"Целевая размерность: {args.umap_dims}")
        logger.info(f"n_neighbors: {args.umap_neighbors}")
        logger.info(f"min_dist: 0.0")
        
        embeddings_reduced, reducer = reduce_dimensions_umap(
            embeddings, 
            n_components=args.umap_dims,
            n_neighbors=args.umap_neighbors,
            sample_size=args.umap_sample
        )
        
        # Сохраняем UMAP reducer
        temp_output_dir = Path(args.output).parent.resolve()
        reducer_path = save_umap_reducer(
            reducer, 
            temp_output_dir, 
            args.umap_dims,
            args.umap_neighbors,
            embeddings.shape[1],
            timestamp
        )
        
        # Добавляем UMAP embeddings в граф
        add_umap_embeddings_to_graph(G, node_ids, embeddings_reduced, args.umap_dims)
    
    # Определяем output_dir для всех путей (нужен и для elbow, и для кластеризации)
    output_dir = Path(args.output).parent.resolve()
    
    # 4. Elbow method или выбор k
    if args.chosen_k:
        logger.info(f"\n⚠️  Используется заданный k={args.chosen_k} (elbow method пропущен)")
        chosen_k = args.chosen_k
        elbow_results = None
    elif args.k_values:
        logger.info("\n" + "="*60)
        logger.info("ELBOW METHOD: поиск оптимального k")
        logger.info("="*60)
        
        k_values = [int(k.strip()) for k in args.k_values.split(',')]
        logger.info(f"Тестируемые значения k: {k_values}")
        
        elbow_results = find_optimal_k_elbow(embeddings_reduced, k_values)
        
        # output_dir уже определена выше после UMAP
        
        # Формируем имена файлов с учётом постфикса
        suffix = args.output_suffix if args.output_suffix else ''
        
        # Сохраняем результаты
        elbow_json_path = output_dir / f'stage2_umap_elbow_results{suffix}.json'
        with open(elbow_json_path, 'w') as f:
            json.dump(elbow_results, f, indent=2)
        logger.info(f"Результаты elbow method сохранены: {elbow_json_path}")
        
        # Строим график
        elbow_plot_path = output_dir / f'stage2_umap_elbow_plot{suffix}.png'
        plot_elbow_curve(elbow_results, elbow_plot_path)
        
        # Автоматический выбор k
        best_result = max(elbow_results, key=lambda x: x['silhouette'])
        chosen_k = best_result['k']
        logger.info(f"\n✅ Рекомендуемый k={chosen_k} (silhouette={best_result['silhouette']:.4f})")
        logger.info("⚠️  Проверьте elbow_plot.png и при необходимости перезапустите с --chosen_k")
        logger.info("\n⚠️  Для кластеризации перезапустите с --chosen_k")
        return
    else:
        raise ValueError("Необходимо указать --k_values или --chosen_k")
    
    # 5. Кластеризация
    logger.info("\n" + "="*60)
    logger.info(f"КЛАСТЕРИЗАЦИЯ с k={chosen_k}")
    logger.info("="*60)
    
    labels, centroids = perform_clustering(embeddings_reduced, chosen_k)
    
    # 6. Вычисление метаданных
    clusters_info = compute_cluster_info(node_ids, embeddings_reduced, labels, centroids)
    
    # 7. Расстояния между кластерами
    nearest_clusters = compute_cluster_distances(centroids)
    
    # 8. Добавление в граф
    add_clusters_to_graph(G, node_ids, labels)
    
    # 9. Сохранение JSON
    output_dir = Path(args.output).parent.resolve()
    suffix = args.output_suffix if args.output_suffix else ''
    json_path = output_dir / f'stage2_umap_semantic_clusters{suffix}.json'
    
    metadata = {
        'method': 'umap_kmeans',
        'umap_dims': args.umap_dims,
        'umap_neighbors': args.umap_neighbors,
        'n_clusters': chosen_k,
        'n_nodes_clustered': len(node_ids),
        'elbow_results': elbow_results,
        'original_dims': embeddings.shape[1],
        'reduced_dims': embeddings_reduced.shape[1]
    }
    
    save_clusters_json(clusters_info, nearest_clusters, G, json_path, metadata)
    
    # 10. Сохранение графа
    logger.info(f"\nСохранение графа с кластерами: {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(G, f)
    
    logger.info("\n" + "="*60)
    logger.info("✅ ЗАВЕРШЕНО")
    logger.info("="*60)
    logger.info("Результаты:")
    logger.info(f"  - Граф с кластерами: {args.output}")
    logger.info(f"  - JSON для разметки: {json_path}")
    if elbow_results:
        suffix = args.output_suffix if args.output_suffix else ''
        logger.info(f"  - Elbow plot: {output_dir / f'stage2_umap_elbow_plot{suffix}.png'}")
    if args.log:
        logger.info(f"  - Лог: {args.log}")


if __name__ == '__main__':
    main()
