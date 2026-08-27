#!/usr/bin/env python3
"""
Stage 2.1 Validation: Оценка качества кластеризации

Метрики:
1. Intra-cluster cohesion (плотность внутри кластера)
2. Inter-cluster separation (разделение между кластерами)
3. Silhouette per cluster (качество каждого кластера)
4. Cluster size balance (равномерность распределения)
5. Visual validation: UMAP 2D projection с центроидами

Визуализация:
- Только центроиды (быстро)
- Центроиды + выборка (medium)
- Density heatmap (красиво)
- Interactive Plotly (исследование)
"""

import pickle
import json
import logging
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import umap

# Настройка научного стиля для публикации
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})


def setup_logging(log_file=None):
    handlers = []
    
    if log_file:
        # Только файл с полным выводом
        handlers.append(logging.FileHandler(log_file, mode='w', encoding='utf-8'))
        handlers.append(logging.StreamHandler())
    else:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def load_graph_with_clusters(path):
    logger = logging.getLogger(__name__)
    logger.info(f"Загрузка графа с кластерами: {path}")
    with open(path, 'rb') as f:
        G = pickle.load(f)
    
    clustered_nodes = [(n, d['semantic_cluster']) for n, d in G.nodes(data=True) 
                       if 'semantic_cluster' in d]
    
    logger.info(f"Граф: {G.number_of_nodes():,} нод, {len(clustered_nodes):,} кластеризовано")
    return G, clustered_nodes


def extract_embeddings_and_labels(G, clustered_nodes, embedding_key='embedding'):
    logger = logging.getLogger(__name__)
    logger.info(f"Извлечение embeddings и labels (ключ: '{embedding_key}')...")
    
    node_ids = []
    embeddings = []
    labels = []
    
    for node_id, cluster_id in tqdm(clustered_nodes, desc="Извлечение"):
        node_data = G.nodes[node_id]
        if embedding_key in node_data:
            node_ids.append(node_id)
            embeddings.append(node_data[embedding_key])
            labels.append(cluster_id)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    logger.info(f"Извлечено {len(embeddings):,} embeddings ({embeddings.shape[1]}D) для {len(set(labels))} кластеров")
    return node_ids, embeddings, labels


def compute_intra_cluster_cohesion(embeddings, labels):
    """Вычисляет среднее расстояние точек до центроида для каждого кластера"""
    logger = logging.getLogger(__name__)
    logger.info("\n1. Intra-cluster cohesion (плотность внутри кластеров)")
    
    cohesion = {}
    unique_labels = np.unique(labels)
    
    for cluster_id in tqdm(unique_labels, desc="Cohesion"):
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        
        centroid = cluster_embeddings.mean(axis=0)
        
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        cohesion[int(cluster_id)] = {
            'mean_distance': float(distances.mean()),
            'std_distance': float(distances.std()),
            'max_distance': float(distances.max()),
            'size': int(mask.sum())
        }
    
    mean_cohesion = np.mean([c['mean_distance'] for c in cohesion.values()])
    logger.info(f"  Средняя cohesion: {mean_cohesion:.4f}")
    
    best_clusters = sorted(cohesion.items(), key=lambda x: x[1]['mean_distance'])[:5]
    logger.info(f"  Топ-5 самых плотных кластеров:")
    for cid, info in best_clusters:
        logger.info(f"    Кластер {cid}: mean_dist={info['mean_distance']:.4f}, size={info['size']}")
    
    return cohesion


def compute_inter_cluster_separation(embeddings, labels):
    """Вычисляет расстояния между центроидами кластеров"""
    logger = logging.getLogger(__name__)
    logger.info("\n2. Inter-cluster separation (разделение между кластерами)")
    
    unique_labels = np.unique(labels)
    centroids = []
    
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    distances = cosine_distances(centroids)
    
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    
    logger.info(f"  Минимальное расстояние между кластерами: {min_distances.min():.4f}")
    logger.info(f"  Среднее расстояние между кластерами: {min_distances.mean():.4f}")
    logger.info(f"  Максимальное расстояние: {min_distances.max():.4f}")
    
    close_pairs = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            if distances[i, j] < 0.2:
                close_pairs.append((int(unique_labels[i]), int(unique_labels[j]), distances[i, j]))
    
    if close_pairs:
        logger.info(f"  ⚠️  {len(close_pairs)} пар кластеров очень близко (distance < 0.2):")
        for c1, c2, dist in sorted(close_pairs, key=lambda x: x[2])[:5]:
            logger.info(f"    Кластеры {c1} и {c2}: distance={dist:.4f}")
    
    return {
        'centroids': centroids,
        'distances': distances,
        'min': float(min_distances.min()),
        'mean': float(min_distances.mean()),
        'max': float(min_distances.max()),
        'close_pairs': close_pairs
    }


def compute_silhouette_per_cluster(embeddings, labels, sample_size=5000):
    """Вычисляет silhouette score для каждого кластера"""
    logger = logging.getLogger(__name__)
    logger.info("\n3. Silhouette per cluster (качество каждого кластера)")
    
    if len(embeddings) > sample_size:
        logger.info(f"  Используем выборку {sample_size:,} из {len(embeddings):,}")
        sample_indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings_sample = embeddings[sample_indices]
        labels_sample = labels[sample_indices]
    else:
        embeddings_sample = embeddings
        labels_sample = labels
    
    logger.info("  Вычисление silhouette scores...")
    silhouette_vals = silhouette_samples(embeddings_sample, labels_sample, metric='euclidean')
    
    cluster_silhouettes = {}
    unique_labels = np.unique(labels_sample)
    
    for cluster_id in unique_labels:
        mask = labels_sample == cluster_id
        cluster_sil = silhouette_vals[mask]
        
        cluster_silhouettes[int(cluster_id)] = {
            'mean': float(cluster_sil.mean()),
            'std': float(cluster_sil.std()),
            'min': float(cluster_sil.min()),
            'max': float(cluster_sil.max())
        }
    
    overall_mean = np.mean([s['mean'] for s in cluster_silhouettes.values()])
    logger.info(f"  Общий средний silhouette: {overall_mean:.4f}")
    
    best = sorted(cluster_silhouettes.items(), key=lambda x: x[1]['mean'], reverse=True)[:5]
    worst = sorted(cluster_silhouettes.items(), key=lambda x: x[1]['mean'])[:5]
    
    logger.info(f"  Топ-5 лучших кластеров:")
    for cid, info in best:
        logger.info(f"    Кластер {cid}: silhouette={info['mean']:.4f}")
    
    logger.info(f"  Топ-5 худших кластеров:")
    for cid, info in worst:
        logger.info(f"    Кластер {cid}: silhouette={info['mean']:.4f}")
    
    return cluster_silhouettes


def check_cluster_size_balance(labels):
    """Проверяет равномерность распределения размеров кластеров"""
    logger = logging.getLogger(__name__)
    logger.info("\n4. Cluster size balance (равномерность распределения)")
    
    cluster_sizes = Counter(labels)
    sizes = list(cluster_sizes.values())
    
    logger.info(f"  Количество кластеров: {len(cluster_sizes)}")
    logger.info(f"  Размер кластеров:")
    logger.info(f"    Min: {min(sizes):,}")
    logger.info(f"    Max: {max(sizes):,}")
    logger.info(f"    Mean: {np.mean(sizes):.1f}")
    logger.info(f"    Median: {np.median(sizes):.1f}")
    logger.info(f"    Std: {np.std(sizes):.1f}")
    
    cv = np.std(sizes) / np.mean(sizes)
    logger.info(f"    Coefficient of variation: {cv:.2f}")
    
    if cv < 0.5:
        logger.info(f"  ✅ Кластеры сбалансированы (CV < 0.5)")
    elif cv < 1.0:
        logger.info(f"  ⚠️  Кластеры умеренно несбалансированы (0.5 < CV < 1.0)")
    else:
        logger.info(f"  ❌ Кластеры сильно несбалансированы (CV > 1.0)")
    
    return {
        'n_clusters': len(cluster_sizes),
        'sizes': dict(cluster_sizes),
        'min': int(min(sizes)),
        'max': int(max(sizes)),
        'mean': float(np.mean(sizes)),
        'median': float(np.median(sizes)),
        'std': float(np.std(sizes)),
        'cv': float(cv)
    }


def plot_centroid_distance_matrix(distances, unique_labels, output_dir, mean_dist=None, median_dist=None):
    """Матрица расстояний между центроидами кластеров (publication quality)"""
    logger = logging.getLogger(__name__)
    logger.info("  Построение матрицы расстояний между центроидами...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Автоматическая подстройка шкалы для UMAP (где расстояния очень малы)
    non_diag = distances[np.triu_indices_from(distances, k=1)]
    max_dist = non_diag.max()
    vmax = 0.5 if max_dist > 0.1 else max_dist * 1.2  # Для UMAP используем адаптивный vmax
    
    logger.info(f"  Colormap range: vmin=0, vmax={vmax:.4f} (max_distance={max_dist:.4f})")
    
    # Научная цветовая схема
    im = ax.imshow(distances, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
    
    # Аннотации только для подвыборки (слишком много для 150x150)
    if len(unique_labels) <= 50:
        ax.set_xticks(range(len(unique_labels)))
        ax.set_yticks(range(len(unique_labels)))
        ax.set_xticklabels(unique_labels, fontsize=7, rotation=90)
        ax.set_yticklabels(unique_labels, fontsize=7)
    else:
        # Для больших матриц - показываем каждый N-й тик
        step = max(1, len(unique_labels) // 20)
        ticks = list(range(0, len(unique_labels), step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([unique_labels[i] for i in ticks], fontsize=8, rotation=90)
        ax.set_yticklabels([unique_labels[i] for i in ticks], fontsize=8)
    
    ax.set_xlabel('Cluster ID', fontweight='bold')
    ax.set_ylabel('Cluster ID', fontweight='bold')
    ax.set_title('Cosine Distance Matrix Between Cluster Centroids', 
                 fontweight='bold', pad=15)
    
    # Colorbar с информативными метками
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Distance', fontweight='bold', rotation=270, labelpad=20)
    
    # Добавляем статистические аннотации
    if mean_dist is not None and median_dist is not None:
        textstr = f'Mean: {mean_dist:.3f}\\n' + f'Median: {median_dist:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'cluster_distance_matrix.png'
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  ✅ Сохранено: {output_path}")


def create_publication_figure(embeddings, labels, centroids, distances, 
                              cluster_silhouettes, cluster_sizes_info,
                              output_dir, sample_size=10000, n_closest_pairs=15):
    """
    Создаёт комбинированную фигуру для публикации с 3 панелями:
    (a) Центроиды в 2D UMAP с метриками
    (b) Матрица расстояний (сжатая)
    (c) Size vs Silhouette scatter
    """
    logger = logging.getLogger(__name__)
    logger.info("\n📊 Создание publication-ready combined figure...")
    
    unique_labels = np.unique(labels)
    cluster_sizes = Counter(labels)
    
    # Вычисляем общие метрики
    total_nodes = len(labels)
    n_clusters = len(unique_labels)
    overall_silhouette = np.mean([s['mean'] for s in cluster_silhouettes.values()])
    sizes_list = list(cluster_sizes.values())
    cv = np.std(sizes_list) / np.mean(sizes_list)
    
    # === Проекция центроидов в 2D ===
    logger.info("  Проецирование центроидов в 2D через UMAP...")
    reducer_centroids = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(centroids)-1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    centroids_2d = reducer_centroids.fit_transform(centroids)
    
    # === Создание multi-panel figure ===
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3, hspace=0.3)
    
    # === PANEL (a): Центроиды с линиями ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Линии между ближайшими парами
    if n_closest_pairs > 0:
        dist_copy = distances.copy()
        np.fill_diagonal(dist_copy, np.inf)
        
        closest_pairs = []
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                closest_pairs.append((i, j, dist_copy[i, j]))
        
        closest_pairs.sort(key=lambda x: x[2])
        
        # Рисуем линии с переменной толщиной
        for i, j, dist in closest_pairs[:n_closest_pairs]:
            width = 0.2 + (1.0 - dist) * 1.5  # Диапазон: 0.2-1.7px
            ax1.plot(
                [centroids_2d[i, 0], centroids_2d[j, 0]],
                [centroids_2d[i, 1], centroids_2d[j, 1]],
                color='gray', linewidth=width, alpha=0.4, zorder=1
            )
    
    # Точки центроидов - цвет по размеру кластера
    sizes_array = np.array([cluster_sizes[label] for label in unique_labels])
    scatter = ax1.scatter(
        centroids_2d[:, 0], 
        centroids_2d[:, 1],
        c=sizes_array,
        s=sizes_array / 5,  # Размер точки пропорционален размеру кластера
        alpha=0.7,
        cmap='plasma',
        edgecolors='black',
        linewidth=0.8,
        zorder=2
    )
    
    # Аннотации с ID и размером (выборочно, чтобы не загромождать)
    annotation_step = max(1, len(unique_labels) // 30)  # Показываем ~30 меток
    for i in range(0, len(centroids_2d), annotation_step):
        x, y = centroids_2d[i]
        size = cluster_sizes[unique_labels[i]]
        ax1.annotate(
            f'{unique_labels[i]}',
            (x, y),
            fontsize=6,
            ha='center',
            va='center',
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5, edgecolor='none')
        )
    
    ax1.set_xlabel('UMAP Dimension 1', fontweight='bold')
    ax1.set_ylabel('UMAP Dimension 2', fontweight='bold')
    ax1.set_title('(a) Cluster Centroids in 2D UMAP Space', fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.2, linewidth=0.5)
    
    # Colorbar для размера
    cbar1 = plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Cluster Size', fontweight='bold', rotation=270, labelpad=15)
    
    # Легенда с метриками
    metrics_text = (f'N = {total_nodes:,} nodes\n' + 
                   f'K = {n_clusters} clusters\n' + 
                   f'Silhouette = {overall_silhouette:.3f}\n' + 
                   f'CV = {cv:.2f}')
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5)
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props, family='monospace')
    
    # === PANEL (b): Матрица расстояний (сжатая версия) ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Автоматическая подстройка шкалы для UMAP (где расстояния очень малы)
    non_diag = distances[np.triu_indices_from(distances, k=1)]
    max_dist_matrix = non_diag.max()
    vmax_matrix = 0.5 if max_dist_matrix > 0.1 else max_dist_matrix * 1.2
    
    # Обрезаем шкалу для контраста
    im = ax2.imshow(distances, cmap='viridis', aspect='auto', vmin=0, vmax=vmax_matrix, interpolation='nearest')
    
    # Убираем подписи осей
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    ax2.set_title('(b) Inter-Cluster Distance Matrix', fontweight='bold', pad=10)
    
    cbar2 = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Cosine Distance', fontweight='bold', rotation=270, labelpad=15)
    
    # Статистики расстояний
    dist_flat = distances[np.triu_indices_from(distances, k=1)]  # Только верхняя треугольная часть
    mean_dist = dist_flat.mean()
    median_dist = np.median(dist_flat)
    
    # Адаптивный формат: для малых расстояний UMAP показываем больше знаков
    if mean_dist < 0.01:
        dist_text = f'Mean: {mean_dist:.4f}\nMedian: {median_dist:.4f}'
    else:
        dist_text = f'Mean: {mean_dist:.3f}\nMedian: {median_dist:.3f}'
    
    ax2.text(0.02, 0.98, dist_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props, family='monospace')
    
    # === PANEL (c): Size vs Silhouette scatter ===
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Подготовка данных
    cluster_ids = []
    sizes_for_scatter = []
    silhouettes_for_scatter = []
    
    for cid in unique_labels:
        if int(cid) in cluster_silhouettes:
            cluster_ids.append(cid)
            sizes_for_scatter.append(cluster_sizes[cid])
            silhouettes_for_scatter.append(cluster_silhouettes[int(cid)]['mean'])
    
    sizes_for_scatter = np.array(sizes_for_scatter)
    silhouettes_for_scatter = np.array(silhouettes_for_scatter)
    
    # Scatter с градиентом цвета
    scatter3 = ax3.scatter(
        sizes_for_scatter,
        silhouettes_for_scatter,
        c=silhouettes_for_scatter,
        s=50,
        alpha=0.6,
        cmap='coolwarm',
        edgecolors='black',
        linewidth=0.5
    )
    
    ax3.set_xlabel('Cluster Size (nodes)', fontweight='bold')
    ax3.set_ylabel('Mean Silhouette Score', fontweight='bold')
    ax3.set_title('(c) Cluster Quality vs Size', fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.2, linewidth=0.5)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    cbar3 = plt.colorbar(scatter3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Silhouette Score', fontweight='bold', rotation=270, labelpad=15)
    
    # Корреляция
    if len(sizes_for_scatter) > 2:
        correlation = np.corrcoef(sizes_for_scatter, silhouettes_for_scatter)[0, 1]
        corr_text = f'r = {correlation:.3f}'
        ax3.text(0.98, 0.02, corr_text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=props, family='monospace')
    
    # Сохранение
    output_path = Path(output_dir) / 'publication_combined_figure.png'
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  ✅ Combined figure сохранена: {output_path}")
    
    return centroids_2d, mean_dist, median_dist


def visualize_clusters_2d(embeddings, labels, centroids, output_dir, 
                          mode='centroids', sample_size=5000, 
                          distances=None, n_closest_pairs=10,
                          cluster_silhouettes=None):
    """
    Визуализация кластеров в 2D через UMAP (улучшенная для публикации)
    
    Режимы:
    - 'centroids': только центроиды
    - 'sample': центроиды + выборка нод
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n5. Визуализация кластеров (режим: {mode})")
    
    logger.info("  Проецирование центроидов на 2D через UMAP...")
    reducer_centroids = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(centroids)-1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    centroids_2d = reducer_centroids.fit_transform(centroids)
    
    unique_labels = np.unique(labels)
    cluster_sizes = Counter(labels)
    
    # Метрики для легенды
    total_nodes = len(labels)
    n_clusters = len(unique_labels)
    if cluster_silhouettes:
        overall_silhouette = np.mean([s['mean'] for s in cluster_silhouettes.values()])
    else:
        overall_silhouette = 0.0
    sizes_list = list(cluster_sizes.values())
    cv = np.std(sizes_list) / np.mean(sizes_list)
    
    if mode == 'centroids':
        logger.info("  Рисуем центроиды (publication quality)...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Линии между ближайшими
        if distances is not None and n_closest_pairs > 0:
            logger.info(f"  Добавление линий между {n_closest_pairs} ближайшими парами...")
            dist_copy = distances.copy()
            np.fill_diagonal(dist_copy, np.inf)
            
            closest_pairs = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    closest_pairs.append((i, j, dist_copy[i, j]))
            
            closest_pairs.sort(key=lambda x: x[2])
            
            # Рисуем линии с переменной толщиной (зависит от расстояния)
            for i, j, dist in closest_pairs[:n_closest_pairs]:
                # Толщина: чем меньше distance, тем толще линия
                # Диапазон: 0.3px (дальние) - 2.5px (ближние)
                width = 0.3 + (1.0 - dist) * 2.2
                ax.plot(
                    [centroids_2d[i, 0], centroids_2d[j, 0]],
                    [centroids_2d[i, 1], centroids_2d[j, 1]],
                    color='gray', linewidth=width, alpha=0.4, zorder=1
                )
        
        # Центроиды - цвет по размеру
        sizes = np.array([cluster_sizes[label] for label in unique_labels])
        
        scatter = ax.scatter(
            centroids_2d[:, 0], 
            centroids_2d[:, 1],
            c=sizes,
            s=sizes / 5,
            alpha=0.7,
            cmap='plasma',
            edgecolors='black',
            linewidth=0.8,
            zorder=2
        )
        
        # Аннотации (выборочно)
        annotation_step = max(1, len(unique_labels) // 40)
        for i in range(0, len(centroids_2d), annotation_step):
            x, y = centroids_2d[i]
            size = cluster_sizes[unique_labels[i]]
            ax.annotate(
                f'{unique_labels[i]}\\n({size})',
                (x, y),
                fontsize=6,
                ha='center',
                va='center',
                fontweight='bold'
            )
        
        ax.set_xlabel('UMAP Dimension 1', fontweight='bold')
        ax.set_ylabel('UMAP Dimension 2', fontweight='bold')
        ax.set_title('Semantic Cluster Centroids (2D UMAP Projection)', 
                     fontweight='bold', pad=15)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax, label='Cluster Size')
        cbar.set_label('Cluster Size', fontweight='bold', rotation=270, labelpad=20)
        
        # Легенда с метриками
        metrics_text = (f'Nodes: {total_nodes:,}\\n' + 
                       f'Clusters: {n_clusters}\\n' + 
                       f'Silhouette: {overall_silhouette:.3f}\\n' + 
                       f'CV: {cv:.2f}')
        props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')
        
        output_path = Path(output_dir) / 'cluster_visualization_centroids.png'
        plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✅ Сохранено: {output_path}")
    
    elif mode == 'sample':
        logger.info(f"  Рисуем центроиды + {sample_size} нод (publication quality)...")
        
        if len(embeddings) > sample_size:
            sample_indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
            embeddings_sample = embeddings[sample_indices]
            labels_sample = labels[sample_indices]
        else:
            embeddings_sample = embeddings
            labels_sample = labels
        
        logger.info("  Трансформируем выборку в 2D...")
        points_2d = reducer_centroids.transform(embeddings_sample)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Точки выборки
        ax.scatter(
            points_2d[:, 0],
            points_2d[:, 1],
            c=labels_sample,
            s=3,
            alpha=0.2,
            cmap='tab20',
            zorder=1
        )
        
        # Линии между ближайшими центроидами
        if distances is not None and n_closest_pairs > 0:
            dist_copy = distances.copy()
            np.fill_diagonal(dist_copy, np.inf)
            
            closest_pairs = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    closest_pairs.append((i, j, dist_copy[i, j]))
            
            closest_pairs.sort(key=lambda x: x[2])
            
            # Рисуем линии с переменной толщиной
            for i, j, dist in closest_pairs[:n_closest_pairs]:
                width = 0.3 + (1.0 - dist) * 1.8  # Диапазон: 0.3-2.1px
                ax.plot(
                    [centroids_2d[i, 0], centroids_2d[j, 0]],
                    [centroids_2d[i, 1], centroids_2d[j, 1]],
                    color='gray', linewidth=width, alpha=0.4, zorder=2
                )
        
        # Центроиды
        sizes = np.array([cluster_sizes[label] for label in unique_labels])
        scatter = ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c=sizes,
            s=sizes / 3,
            alpha=0.9,
            cmap='plasma',
            edgecolors='black',
            linewidth=1.2,
            marker='*',
            zorder=3
        )
        
        # Аннотации (выборочно)
        annotation_step = max(1, len(unique_labels) // 30)
        for i in range(0, len(centroids_2d), annotation_step):
            x, y = centroids_2d[i]
            ax.annotate(
                f'{unique_labels[i]}',
                (x, y),
                fontsize=6,
                ha='center',
                va='center',
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none')
            )
        
        ax.set_xlabel('UMAP Dimension 1', fontweight='bold')
        ax.set_ylabel('UMAP Dimension 2', fontweight='bold')
        ax.set_title(f'Cluster Visualization ({sample_size:,} samples + centroids)',
                     fontweight='bold', pad=15)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster Size', fontweight='bold', rotation=270, labelpad=20)
        
        # Легенда с метриками
        metrics_text = (f'Nodes: {total_nodes:,}\\n' + 
                       f'Clusters: {n_clusters}\\n' + 
                       f'Silhouette: {overall_silhouette:.3f}\\n' + 
                       f'CV: {cv:.2f}')
        props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')
        
        output_path = Path(output_dir) / 'cluster_visualization_sample.png'
        plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✅ Сохранено: {output_path}")
    
    return centroids_2d
    """
    Визуализация кластеров в 2D через UMAP
    
    Режимы:
    - 'centroids': только центроиды
    - 'sample': центроиды + выборка нод
    - 'density': тепловая карта плотности
    
    Опционально рисует линии между n_closest_pairs ближайших центроидов
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n5. Визуализация кластеров (режим: {mode})")
    
    logger.info("  Проецирование центроидов на 2D через UMAP...")
    reducer_centroids = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(centroids)-1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    centroids_2d = reducer_centroids.fit_transform(centroids)
    
    unique_labels = np.unique(labels)
    cluster_sizes = Counter(labels)
    
    if mode == 'centroids':
        logger.info("  Рисуем только центроиды...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        sizes = [cluster_sizes[label] for label in unique_labels]
        
        scatter = ax.scatter(
            centroids_2d[:, 0], 
            centroids_2d[:, 1],
            c=unique_labels,
            s=np.array(sizes) / 10,
            alpha=0.6,
            cmap='tab20',
            edgecolors='black',
            linewidth=1
        )
        
        if distances is not None and n_closest_pairs > 0:
            logger.info(f"  Добавление линий между {n_closest_pairs} ближайшими парами...")
            dist_copy = distances.copy()
            np.fill_diagonal(dist_copy, np.inf)
            
            closest_pairs = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    closest_pairs.append((i, j, dist_copy[i, j]))
            
            closest_pairs.sort(key=lambda x: x[2])
            
            # Рисуем линии с переменной толщиной
            for i, j, dist in closest_pairs[:n_closest_pairs]:
                width = 0.3 + (1.0 - dist) * 2.2  # Диапазон: 0.3-2.5px
                ax.plot(
                    [centroids_2d[i, 0], centroids_2d[j, 0]],
                    [centroids_2d[i, 1], centroids_2d[j, 1]],
                    color='gray', linewidth=width, alpha=0.5, zorder=1
                )
        
        for i, (x, y) in enumerate(centroids_2d):
            size = cluster_sizes[unique_labels[i]]
            ax.annotate(
                f'{unique_labels[i]}\n({size})',
                (x, y),
                fontsize=7,
                ha='center',
                va='center',
                fontweight='bold'
            )
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
        ax.set_title('Cluster Centroids (2D UMAP projection)\\n' + 
                     'Size = cluster size, gray lines = closest pairs', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, label='Cluster ID')
        
        output_path = Path(output_dir) / 'cluster_visualization_centroids.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✅ Сохранено: {output_path}")
    
    elif mode == 'sample':
        logger.info(f"  Рисуем центроиды + {sample_size} случайных нод...")
        
        if len(embeddings) > sample_size:
            sample_indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
            embeddings_sample = embeddings[sample_indices]
            labels_sample = labels[sample_indices]
        else:
            embeddings_sample = embeddings
            labels_sample = labels
        
        logger.info("  Трансформируем выборку в 2D...")
        points_2d = reducer_centroids.transform(embeddings_sample)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        ax.scatter(
            points_2d[:, 0],
            points_2d[:, 1],
            c=labels_sample,
            s=5,
            alpha=0.3,
            cmap='tab20'
        )
        
        sizes = [cluster_sizes[label] for label in unique_labels]
        scatter = ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c=unique_labels,
            s=np.array(sizes) / 5,
            alpha=0.8,
            cmap='tab20',
            edgecolors='black',
            linewidth=2,
            marker='*'
        )
        
        if distances is not None and n_closest_pairs > 0:
            logger.info(f"  Добавление линий между {n_closest_pairs} ближайшими парами...")
            dist_copy = distances.copy()
            np.fill_diagonal(dist_copy, np.inf)
            
            closest_pairs = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    closest_pairs.append((i, j, dist_copy[i, j]))
            
            closest_pairs.sort(key=lambda x: x[2])
            
            # Рисуем линии с переменной толщиной
            for i, j, dist in closest_pairs[:n_closest_pairs]:
                width = 0.3 + (1.0 - dist) * 2.2  # Диапазон: 0.3-2.5px
                ax.plot(
                    [centroids_2d[i, 0], centroids_2d[j, 0]],
                    [centroids_2d[i, 1], centroids_2d[j, 1]],
                    color='gray', linewidth=width, alpha=0.4, zorder=1
                )
        
        for i, (x, y) in enumerate(centroids_2d):
            size = cluster_sizes[unique_labels[i]]
            ax.annotate(
                f'{unique_labels[i]}\n({size})',
                (x, y),
                fontsize=7,
                ha='center',
                va='center',
                fontweight='bold'
            )
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
        ax.set_title(f'Cluster Visualization: {sample_size} samples + centroids\\n' + 
                     '(2D UMAP projection, gray lines = closest pairs)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        output_path = Path(output_dir) / 'cluster_visualization_sample.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  ✅ Сохранено: {output_path}")
    
    return centroids_2d


def plot_size_distribution(cluster_sizes, output_dir, cluster_silhouettes=None):
    """Гистограмма распределения размеров кластеров (улучшенная для публикации)"""
    logger = logging.getLogger(__name__)
    logger.info("  Построение гистограммы размеров...")
    
    sizes = list(cluster_sizes['sizes'].values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Левая панель: гистограмма
    ax1.hist(sizes, bins=30, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
    ax1.axvline(cluster_sizes['mean'], color='#e74c3c', linestyle='--', linewidth=2, 
                label=f"Mean: {cluster_sizes['mean']:.0f}")
    ax1.axvline(cluster_sizes['median'], color='#2ecc71', linestyle='--', linewidth=2, 
                label=f"Median: {cluster_sizes['median']:.0f}")
    ax1.set_xlabel('Cluster Size (nodes)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('(a) Cluster Size Distribution', fontweight='bold', pad=10)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.2, linewidth=0.5, axis='y')
    
    # Правая панель: упорядоченный bar plot
    sorted_sizes = sorted(cluster_sizes['sizes'].items(), key=lambda x: x[1], reverse=True)
    cluster_ids = [c[0] for c in sorted_sizes]
    sizes_sorted = [c[1] for c in sorted_sizes]
    
    # Цвет по размеру
    colors = plt.cm.plasma(np.linspace(0, 1, len(sizes_sorted)))
    
    ax2.bar(range(len(sizes_sorted)), sizes_sorted, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(cluster_sizes['mean'], color='#e74c3c', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f"Mean: {cluster_sizes['mean']:.0f}")
    ax2.set_xlabel('Cluster Rank (by size)', fontweight='bold')
    ax2.set_ylabel('Size (nodes)', fontweight='bold')
    ax2.set_title('(b) Clusters Ranked by Size', fontweight='bold', pad=10)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.2, linewidth=0.5, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'cluster_size_distribution.png'
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"  ✅ Сохранено: {output_path}")


def convert_numpy_types(obj):
    """Рекурсивно конвертирует numpy типы в стандартные python типы"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj


def save_validation_report(metrics, output_path):
    """Сохраняет JSON отчёт со всеми метриками"""
    logger = logging.getLogger(__name__)
    logger.info(f"\nСохранение отчёта: {output_path}")
    
    # Конвертируем numpy типы перед сохранением
    metrics_clean = convert_numpy_types(metrics)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_clean, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Отчёт сохранён")


def main():
    base_dir = Path(__file__).parent
    
    parser = argparse.ArgumentParser(description='Stage 2.1 Validation: оценка качества кластеризации')
    parser.add_argument('--graph_path', 
                        default=str(base_dir / 'stage2.1/results/step2_models/graph_stage2_umap_k40.pkl'),
                        help='Путь к графу с кластерами')
    parser.add_argument('--output_dir', 
                        default=str(base_dir / 'stage2.1/results/step4_analysis'),
                        help='Директория для сохранения результатов')
    parser.add_argument('--embedding-key', dest='embedding_key', default='embedding',
                        help='Ключ embedding в node data (default: "embedding" для 1536D, "embedding_umap_150d" для UMAP)')
    parser.add_argument('--viz_mode', choices=['centroids', 'sample', 'both'], default='both',
                        help='Режим визуализации')
    parser.add_argument('--sample_size', type=int, default=5000, help='Размер выборки для viz')
    parser.add_argument('--log', type=str, help='Путь к лог файлу')
    
    args = parser.parse_args()
    
    logger = setup_logging(args.log)
    
    logger.info("="*60)
    logger.info("STAGE 2.1 VALIDATION: Оценка качества кластеризации")
    logger.info("="*60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Создаем подпапки для новой структуры
    metrics_dir = output_dir / 'metrics'
    figures_dir = output_dir / 'figures'
    metrics_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    G, clustered_nodes = load_graph_with_clusters(args.graph_path)
    
    node_ids, embeddings, labels = extract_embeddings_and_labels(G, clustered_nodes, args.embedding_key)
    
    metrics = {}
    
    metrics['cohesion'] = compute_intra_cluster_cohesion(embeddings, labels)
    
    separation = compute_inter_cluster_separation(embeddings, labels)
    metrics['separation'] = {
        'min': separation['min'],
        'mean': separation['mean'],
        'max': separation['max'],
        'close_pairs': separation['close_pairs']
    }
    centroids = separation['centroids']
    
    metrics['silhouette_per_cluster'] = compute_silhouette_per_cluster(embeddings, labels)
    
    # Добавляем Calinski-Harabasz и Davies-Bouldin
    logger.info("\nДополнительные метрики качества...")
    metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(embeddings, labels))
    metrics['davies_bouldin_score'] = float(davies_bouldin_score(embeddings, labels))
    logger.info(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
    logger.info(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.2f}")
    
    cluster_sizes = check_cluster_size_balance(labels)
    metrics['cluster_sizes'] = cluster_sizes
    
    # === ПУБЛИКАЦИОННАЯ ВИЗУАЛИЗАЦИЯ ===
    
    # 1. Combined figure (главная фигура для публикации)
    logger.info("\n" + "="*60)
    logger.info("СОЗДАНИЕ PUBLICATION-READY ВИЗУАЛИЗАЦИЙ")
    logger.info("="*60)
    
    centroids_2d, mean_dist, median_dist = create_publication_figure(
        embeddings, labels, centroids, 
        separation['distances'],
        metrics['silhouette_per_cluster'],
        cluster_sizes,
        figures_dir,  # Используем figures_dir
        sample_size=args.sample_size,
        n_closest_pairs=20
    )
    
    # 2. Отдельные высококачественные визуализации
    plot_size_distribution(cluster_sizes, figures_dir, metrics['silhouette_per_cluster'])
    
    plot_centroid_distance_matrix(separation['distances'], np.unique(labels), 
                                   figures_dir, mean_dist, median_dist)
    
    if args.viz_mode in ['centroids', 'both']:
        visualize_clusters_2d(embeddings, labels, centroids, figures_dir, mode='centroids',
                             distances=separation['distances'], n_closest_pairs=30,
                             cluster_silhouettes=metrics['silhouette_per_cluster'])
    
    report_path = metrics_dir / 'validation_report.json'  # Используем metrics_dir
    save_validation_report(metrics, report_path)
    
    logger.info("\n" + "="*60)
    logger.info("✅ ВАЛИДАЦИЯ ЗАВЕРШЕНА")
    logger.info("="*60)
    logger.info(f"Результаты сохранены в: {output_dir}")


if __name__ == '__main__':
    main()
