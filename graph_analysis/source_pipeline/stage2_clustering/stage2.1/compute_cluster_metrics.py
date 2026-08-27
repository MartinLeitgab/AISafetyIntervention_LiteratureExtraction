#!/usr/bin/env python3
"""
Универсальный скрипт для расчёта метрик кластеров

Вычисляет для любого набора кластеров:
- Graph metrics: degree, betweenness, closeness, PageRank, eigenvector centrality
- Cluster roles: roots (фундаментальные), bridges (мосты), leaves (специализированные), regular
- Communities: Louvain community detection
- Summary reports: текстовые и JSON

Использование:
    python compute_cluster_metrics.py --clusters_json path/to/clusters.json \
                                      --names_json path/to/names.json \
                                      --graph_pkl path/to/graph.pkl \
                                      --output_dir results/
"""

import json
import pickle
import argparse
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects
import numpy as np
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
    print("⚠️ Warning: python-louvain not installed. Community detection might fail.")


def load_data(clusters_path, names_path, graph_path):
    """Загружает кластеры, названия и граф"""
    print("📚 Загрузка данных...")
    
    with open(clusters_path) as f:
        clusters_data = json.load(f)
    
    with open(names_path) as f:
        names_data = json.load(f)
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"  ✅ Кластеров: {len(clusters_data.get('clusters', []))}")
    print(f"  ✅ Нод в графе: {G.number_of_nodes():,}")
    print(f"  ✅ Рёбер в графе: {G.number_of_edges():,}")
    
    return clusters_data, names_data, G


def build_cluster_graph(clusters_data, names_data, top_k_neighbors=5, min_weight=0.6, embedding_key='embedding'):
    """
    Строит граф связей между кластерами
    
    Args:
        clusters_data: данные кластеров с nearest_clusters
        names_data: названия и описания кластеров
        top_k_neighbors: сколько ближайших соседей учитывать
        min_weight: минимальный вес связи (0-1)
        embedding_key: ключ для embeddings ('embedding' или 'embedding_umap_150d')
    
    Returns:
        NetworkX Graph с кластерами как нодами
    """
    print(f"\n🕸️ Построение графа кластеров...")
    print(f"  Параметры: top_k={top_k_neighbors}, min_weight={min_weight}, embedding_key={embedding_key}")
    
    CG = nx.Graph()
    
    # Добавляем ноды
    clusters = clusters_data.get('clusters', [])
    for cluster in tqdm(clusters, desc="Добавление нод"):
        cid = cluster['cluster_id']
        
        # Handle different name formats
        if 'cluster_names' in names_data:
             name = names_data['cluster_names'][str(cid)]['name']
             description = names_data['cluster_names'][str(cid)].get('description', '')
        else:
             # Simple format {"0": "Name"}
             name = names_data.get(str(cid), f"Cluster {cid}")
             description = ""
        
        CG.add_node(
            cid,
            name=name,
            description=description,
            size=cluster['size'],
            centroid=cluster.get('centroid_embedding')
        )
    
    # Добавляем рёбра
    for cluster in tqdm(clusters, desc="Добавление рёбер"):
        cid = cluster['cluster_id']
        
        for neighbor in cluster.get('nearest_clusters', [])[:top_k_neighbors]:
            neighbor_id = neighbor['cluster_id']
            distance = neighbor['distance']
            weight = max(0.001, 1.0 - distance)
            
            if weight >= min_weight and not CG.has_edge(cid, neighbor_id):
                CG.add_edge(cid, neighbor_id, weight=weight, distance=distance)
    
    print(f"  ✅ Граф: {CG.number_of_nodes()} нод, {CG.number_of_edges()} рёбер")
    
    return CG


def compute_graph_metrics(CG):
    """
    Вычисляет все graph centrality метрики
    
    Returns:
        dict: {node_id: {metric_name: value}}
    """
    print("\n📊 Вычисление graph metrics...")
    
    metrics = {}
    
    # 1. Degree centrality
    print("  → Degree centrality...")
    degree_cent = nx.degree_centrality(CG)
    
    # 2. Betweenness centrality
    print("  → Betweenness centrality...")
    betweenness_cent = nx.betweenness_centrality(CG, weight='distance')
    
    # 3. Closeness centrality
    print("  → Closeness centrality...")
    closeness_cent = nx.closeness_centrality(CG, distance='distance')
    
    # 4. PageRank
    print("  → PageRank...")
    pagerank = nx.pagerank(CG, weight='weight')
    
    # 5. Eigenvector centrality
    print("  → Eigenvector centrality...")
    try:
        eigenvector_cent = nx.eigenvector_centrality(CG, weight='weight', max_iter=1000)
    except:
        eigenvector_cent = {node: 0 for node in CG.nodes()}
        print("    ⚠️  Eigenvector не сошёлся, заполнено нулями")
    
    # Объединяем в один dict
    for node in CG.nodes():
        metrics[node] = {
            'degree': CG.degree(node),
            'degree_centrality': degree_cent[node],
            'betweenness_centrality': betweenness_cent[node],
            'closeness_centrality': closeness_cent[node],
            'eigenvector_centrality': eigenvector_cent[node],
            'pagerank': pagerank[node],
        }
    
    print("  ✅ Метрики вычислены")
    
    return metrics


def classify_cluster_roles(CG, metrics):
    """
    Классифицирует кластеры по ролям на основе метрик
    
    Роли:
        - root: топ-15% по importance + минимум 4 связи (фундаментальные темы)
        - bridge: высокая betweenness (мосты между областями)
        - leaf: мало связей или низкая важность (специализированные)
        - regular: остальные
    
    Args:
        CG: граф кластеров
        metrics: dict с метриками
    
    Returns:
        dict: {node_id: {'role': str, 'importance': float, **metrics}}
    """
    print("\n🎯 Классификация ролей кластеров...")
    
    roles = {}
    
    # Вычисляем комбинированную важность
    importance_scores = {}
    for node in CG.nodes():
        m = metrics[node]
        importance = (
            m['pagerank'] * 3.0 +
            m['eigenvector_centrality'] * 2.0 +
            m['degree_centrality'] * 1.0
        )
        importance_scores[node] = importance
    
    # Находим пороги
    sorted_importance = sorted(importance_scores.values(), reverse=True)
    n = len(sorted_importance)
    
    root_threshold = sorted_importance[int(n * 0.15)] if n > 10 else sorted_importance[0]
    bridge_threshold = sorted_importance[int(n * 0.35)] if n > 10 else sorted_importance[-1]
    
    # Классифицируем
    for node in CG.nodes():
        m = metrics[node]
        importance = importance_scores[node]
        
        if importance >= root_threshold and CG.degree(node) >= 4:
            role = 'root'
        elif m['betweenness_centrality'] > 0.02 or \
             (importance >= bridge_threshold and m['betweenness_centrality'] > 0.01):
            role = 'bridge'
        elif CG.degree(node) <= 3 or importance < bridge_threshold:
            role = 'leaf'
        else:
            role = 'regular'
        
        roles[node] = {
            'role': role,
            'importance': importance,
            **m
        }
    
    # Статистика
    role_counts = Counter(r['role'] for r in roles.values())
    print(f"\n  📊 Распределение ролей:")
    print(f"    🌱 Roots (фундаментальные): {role_counts['root']}")
    print(f"    🌉 Bridges (мосты): {role_counts['bridge']}")
    print(f"    🍃 Leaves (специализированные): {role_counts['leaf']}")
    print(f"    📄 Regular (стандартные): {role_counts['regular']}")
    
    return roles


def detect_communities(CG):
    """
    Находит сообщества в графе кластеров (Louvain)
    
    Returns:
        tuple: (communities list, node_to_community dict)
    """
    print("\n🔍 Поиск сообществ (Louvain)...")
    
    communities = nx.community.louvain_communities(CG, weight='weight', seed=42)
    
    node_to_community = {}
    for comm_id, community in enumerate(communities):
        for node in community:
            node_to_community[node] = comm_id
    
    print(f"  ✅ Найдено {len(communities)} сообществ")
    
    return communities, node_to_community


def plot_cluster_network_with_roles(CG, roles, output_dir):
    """
    Рисует граф кластеров, раскрашенный по ролям
    """
    print("\n🎨 Рисуем граф с ролями...")
    plt.figure(figsize=(15, 12))
    
    # Layout
    pos = nx.kamada_kawai_layout(CG, weight='weight')
    
    # Colors
    role_colors = {
        'root': '#2ecc71',    # Green
        'bridge': '#e67e22',  # Orange
        'leaf': '#3498db',    # Blue
        'regular': '#95a5a6'  # Gray
    }
    
    node_colors = [role_colors[roles[n]['role']] for n in CG.nodes()]
    node_sizes = [CG.nodes[n]['size'] / 5 for n in CG.nodes()] # Scale size
    
    # Draw edges (thicker and more visible)
    weights = [CG[u][v]['weight'] for u, v in CG.edges()]
    # Normalize weights for width
    max_weight = max(weights) if weights else 1
    widths = [(w / max_weight) * 4 for w in weights]  # Умеренная толщина
    
    nx.draw_networkx_edges(CG, pos, alpha=0.4, width=widths, edge_color='#7f8c8d')
    
    # Draw nodes
    nx.draw_networkx_nodes(CG, pos, node_color=node_colors, node_size=node_sizes, 
                           alpha=0.9, edgecolors='white', linewidths=1.5)
    
    # Labels
    labels = {n: str(n) for n in CG.nodes()}
    nx.draw_networkx_labels(CG, pos, labels=labels, font_size=8, font_weight='bold')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Roots (Foundational)',
                   markerfacecolor=role_colors['root'], markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Bridges (Connecting)',
                   markerfacecolor=role_colors['bridge'], markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Leaves (Specialized)',
                   markerfacecolor=role_colors['leaf'], markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Regular',
                   markerfacecolor=role_colors['regular'], markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Cluster Network Structure (Roles)\n" + 
              "Edges: Top-5 Nearest Cluster Centroids", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    output_path = output_dir / 'network_graph_with_roles.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Сохранено: {output_path}")


def plot_community_structure(CG, communities, roles, output_dir):
    """
    Рисует граф с сообществами (только цвета, без форм)
    """
    print("\n🎨 Рисуем структуру сообществ...")
    plt.figure(figsize=(15, 12))
    
    # Layout (same as above for consistency)
    pos = nx.kamada_kawai_layout(CG, weight='weight')
    
    # Community colors
    num_communities = len(communities)
    # Используем качественную палитру
    cmap = plt.cm.get_cmap('Set2', num_communities) if num_communities <= 8 else plt.cm.get_cmap('tab20', num_communities)
    
    # Node to community map
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i;
    
    # Draw edges
    weights = [CG[u][v]['weight'] for u, v in CG.edges()]
    max_weight = max(weights) if weights else 1
    widths = [(w / max_weight) * 3 for w in weights]  # Умеренная толщина
    nx.draw_networkx_edges(CG, pos, alpha=0.25, width=widths, edge_color='#95a5a6')
    
    # Draw nodes by community
    for i in range(num_communities):
        nodelist = [n for n in CG.nodes() if node_to_comm[n] == i]
        if not nodelist:
            continue;
            
        sizes = [CG.nodes[n]['size'] / 5 for n in nodelist];
        
        nx.draw_networkx_nodes(CG, pos, nodelist=nodelist, node_color=[cmap(i)], 
                               node_shape='o', node_size=sizes, alpha=0.9, 
                               edgecolors='white', linewidths=1.5);
    
    # Labels
    labels = {n: str(n) for n in CG.nodes()};
    nx.draw_networkx_labels(CG, pos, labels=labels, font_size=8, font_weight='bold');
    
    # Legend with placeholders for manual naming
    legend_elements = [];
    for i in range(num_communities):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label=f'Group {i+1}', 
                       markerfacecolor=cmap(i), markersize=10, markeredgecolor='white')
        );
        
    plt.legend(handles=legend_elements, loc='upper right', title="Macro-Topics");
    
    plt.title(f"Community Structure ({num_communities} communities)\n" + 
              "Edges: Top-5 Nearest Cluster Centroids", fontsize=14, fontweight='bold');
    plt.axis('off');
    
    output_path = output_dir / 'community_structure.png';
    plt.savefig(output_path, dpi=300, bbox_inches='tight');
    plt.close();
    print(f"  ✅ Сохранено: {output_path}")

def plot_opportunity_matrix(CG, roles, output_dir):
    """
    Рисует матрицу возможностей (PageRank vs Size)
    """
    print("\n🎨 Рисуем матрицу возможностей...")
    plt.figure(figsize=(14, 11))
    
    # Data preparation
    pageranks = [roles[n]['pagerank'] for n in CG.nodes()]
    sizes = [CG.nodes[n]['size'] for n in CG.nodes()]
    names = [CG.nodes[n]['name'] for n in CG.nodes()]
    role_list = [roles[n]['role'] for n in CG.nodes()]
    
    # Colors
    role_colors = {
        'root': '#2ecc71',
        'bridge': '#e67e22',
        'leaf': '#3498db',
        'regular': '#95a5a6'
    }
    colors = [role_colors[r] for r in role_list]
    
    # Scatter plot
    plt.scatter(pageranks, sizes, c=colors, alpha=0.7, s=120, edgecolors='w', linewidths=1.5, zorder=2)
    
    # Labels for ALL points with better positioning
    texts = []
    for i, n in enumerate(CG.nodes()):
        t = plt.text(pageranks[i], sizes[i], str(n), 
                    fontsize=8, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='gray', alpha=0.8, linewidth=0.5),
                    zorder=3)
        texts.append(t)

    plt.xlabel('Influence (PageRank)', fontsize=12, fontweight='bold')
    plt.ylabel('Volume (Cluster Size)', fontsize=12, fontweight='bold')
    plt.title('Opportunity Matrix: Influence vs Volume', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, zorder=1)
    
    # Quadrants
    # Calculate medians
    pr_median = np.median(pageranks)
    sz_median = np.median(sizes)
    
    plt.axvline(pr_median, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    plt.axhline(sz_median, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Annotate Quadrants
    plt.text(max(pageranks)*0.9, max(sizes)*0.9, 'Mainstream\\n(High Inf, High Vol)', 
             ha='right', va='top', fontsize=10, fontweight='bold', color='gray', alpha=0.7)
    plt.text(min(pageranks)*1.1, max(sizes)*0.9, 'Popular but Niche\\n(Low Inf, High Vol)', 
             ha='left', va='top', fontsize=10, fontweight='bold', color='gray', alpha=0.7)
    plt.text(max(pageranks)*0.9, min(sizes)*1.1, 'Emerging/Foundational\\n(High Inf, Low Vol)', 
             ha='right', va='bottom', fontsize=10, fontweight='bold', color='gray', alpha=0.7)
    plt.text(min(pageranks)*1.1, min(sizes)*1.1, 'Gaps/Specialized\\n(Low Inf, Low Vol)', 
             ha='left', va='bottom', fontsize=10, fontweight='bold', color='gray', alpha=0.7)

    output_path = output_dir / 'opportunity_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Сохранено: {output_path}")


def plot_importance_ranking(CG, roles, output_dir, top_k=7, bottom_k=7):
    """
    Рисует топ и низ кластеров по PageRank (importance)
    """
    print("\n🎨 Рисуем рейтинг важности (топ + низ)...")
    
    # Prepare data
    data = []
    for node in CG.nodes():
        data.append({
            'id': node,
            'name': CG.nodes[node]['name'],
            'pagerank': roles[node]['pagerank'],
            'importance': roles[node]['importance'],
            'role': roles[node]['role'],
            'size': CG.nodes[node]['size']
        })
    
    # Sort by importance (комбинированная метрика)
    data.sort(key=lambda x: x['importance'], reverse=True)
    
    # Берем топ и низ
    top_data = data[:top_k]
    bottom_data = data[-bottom_k:]
    bottom_data.reverse()  # Самые низкие внизу
    
    # Объединяем: топ + разделитель + низ
    combined_data = top_data + bottom_data
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    names = []
    for d in combined_data:
        name = f"{d['id']}: {d['name'][:45]}..." if len(d['name']) > 45 else f"{d['id']}: {d['name']}"
        names.append(name)
    
    values = [d['importance'] for d in combined_data]
    
    role_colors = {
        'root': '#2ecc71',
        'bridge': '#e67e22',
        'leaf': '#3498db',
        'regular': '#95a5a6'
    }
    colors = [role_colors[d['role']] for d in combined_data]
    
    bars = ax.barh(range(len(combined_data)), values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(combined_data)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Топ вверху
    
    ax.set_xlabel('Importance Score (PageRank × 3 + Eigenvector × 2 + Degree)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_k} & Bottom {bottom_k} Clusters by Importance', fontsize=14, fontweight='bold')
    
    # Добавляем горизонтальную линию разделения между топ и низом
    ax.axhline(y=top_k - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Top/Bottom separator')
    
    # Add size annotations
    for i, bar in enumerate(bars):
        size = combined_data[i]['size']
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                f" n={size:,}", va='center', fontsize=9, color='black', fontweight='bold')
    
    # Add importance values at the start of bars
    for i, bar in enumerate(bars):
        importance = combined_data[i]['importance']
        ax.text(0.001, bar.get_y() + bar.get_height()/2, 
                f"{importance:.4f}", va='center', ha='left', fontsize=8, color='white', fontweight='bold')
                 
    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color=role_colors['root'], label='Root', edgecolor='black', linewidth=0.5),
        plt.Rectangle((0,0),1,1, color=role_colors['bridge'], label='Bridge', edgecolor='black', linewidth=0.5),
        plt.Rectangle((0,0),1,1, color=role_colors['leaf'], label='Leaf', edgecolor='black', linewidth=0.5),
        plt.Rectangle((0,0),1,1, color=role_colors['regular'], label='Regular', edgecolor='black', linewidth=0.5)
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)
    
    ax.grid(axis='x', alpha=0.3)
    
    output_path = output_dir / 'importance_ranking.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Сохранено: {output_path}")


def save_analysis_json(CG, roles, node_to_community, output_path):
    """Сохраняет полный анализ в JSON"""
    print(f"\n💾 Сохранение анализа...")
    
    analysis = {
        'metadata': {
            'total_clusters': CG.number_of_nodes(),
            'total_connections': CG.number_of_edges(),
            'communities': len(set(node_to_community.values()))
        },
        'clusters': []
    }
    
    for node, data in CG.nodes(data=True):
        role_info = roles[node]
        
        cluster_info = {
            'cluster_id': node,
            'name': data['name'],
            'description': data['description'],
            'size': data['size'],
            'role': role_info['role'],
            'community': node_to_community[node],
            'metrics': {
                'degree': role_info['degree'],
                'degree_centrality': role_info['degree_centrality'],
                'betweenness_centrality': role_info['betweenness_centrality'],
                'closeness_centrality': role_info['closeness_centrality'],
                'eigenvector_centrality': role_info['eigenvector_centrality'],
                'pagerank': role_info['pagerank'],
                'importance': role_info['importance']
            },
            'neighbors': [
                {'cluster_id': neighbor, 'weight': CG[node][neighbor]['weight']}
                for neighbor in CG.neighbors(node)
            ]
        }
        
        analysis['clusters'].append(cluster_info)
    
    # Сортируем по PageRank
    analysis['clusters'].sort(key=lambda x: x['metrics']['pagerank'], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ JSON сохранён: {output_path}")


def save_text_report(CG, roles, output_path):
    """Сохраняет текстовый отчёт с топ кластерами по категориям"""
    print(f"\n📝 Создание текстового отчёта...")
    
    lines = []
    
    # Заголовок
    lines.append("=" * 80)
    lines.append("CLUSTER ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Группируем по ролям
    clusters_by_role = defaultdict(list)
    for node in CG.nodes():
        role = roles[node]['role']
        clusters_by_role[role].append((
            node,
            CG.nodes[node]['name'],
            CG.nodes[node]['size'],
            roles[node]
        ))
    
    # ROOTS
    lines.append("=" * 80)
    lines.append("🌱 ROOTS - FUNDAMENTAL TOPICS")
    lines.append("=" * 80)
    roots = sorted(clusters_by_role['root'], 
                   key=lambda x: x[3]['pagerank'], reverse=True)
    lines.append(f"Total: {len(roots)}\n")
    
    for i, (cid, name, size, metrics) in enumerate(roots, 1):
        lines.append(f"{i:2d}. {name}")
        lines.append(f"    PageRank: {metrics['pagerank']:.4f} | "
                    f"Degree: {metrics['degree']} | "
                    f"Size: {size:,} nodes")
        lines.append(f"    {CG.nodes[cid]['description'][:100]}...")
        lines.append("")
    
    # BRIDGES
    lines.append("\n" + "=" * 80)
    lines.append("🌉 BRIDGES - CONNECTING TOPICS")
    lines.append("=" * 80)
    bridges = sorted(clusters_by_role['bridge'],
                    key=lambda x: x[3]['betweenness_centrality'], reverse=True)
    lines.append(f"Total: {len(bridges)}\n")
    
    for i, (cid, name, size, metrics) in enumerate(bridges[:20], 1):
        lines.append(f"{i:2d}. {name}")
        lines.append(f"    Betweenness: {metrics['betweenness_centrality']:.4f} | "
                    f"Degree: {metrics['degree']} | "
                    f"Size: {size:,}")
        lines.append(f"    {CG.nodes[cid]['description'][:100]}...")
        lines.append("")
    
    # REGULAR
    lines.append("\n" + "=" * 80)
    lines.append("📄 REGULAR - STANDARD TOPICS")
    lines.append("=" * 80)
    regular = sorted(clusters_by_role['regular'],
                    key=lambda x: x[3]['pagerank'], reverse=True)
    lines.append(f"Total: {len(regular)}\n")
    
    for i, (cid, name, size, metrics) in enumerate(regular[:15], 1):
        lines.append(f"{i:2d}. {name} "
                    f"(PR: {metrics['pagerank']:.4f}, "
                    f"degree: {metrics['degree']}, "
                    f"size: {size:,})")
    
    # LEAVES
    lines.append("\n\n" + "=" * 80)
    lines.append("🍃 LEAVES - SPECIALIZED TOPICS")
    lines.append("=" * 80)
    leaves = sorted(clusters_by_role['leaf'],
                   key=lambda x: x[2], reverse=True)
    lines.append(f"Total: {len(leaves)}\n")
    
    # Разделим на большие и маленькие
    big_leaves = [l for l in leaves if l[2] > 1000]
    small_leaves = [l for l in leaves if l[2] <= 200]
    
    lines.append("📊 LARGE leaves (>1000 nodes):")
    for i, (cid, name, size, metrics) in enumerate(big_leaves[:15], 1):
        lines.append(f"{i:2d}. {name} ({size:,} nodes, {metrics['degree']} connections)")
    
    lines.append(f"\n🔬 SMALL leaves (≤200 nodes, potential research gaps):")
    for i, (cid, name, size, metrics) in enumerate(small_leaves[:15], 1):
        lines.append(f"{i:2d}. {name} ({size} nodes, {metrics['degree']} connections)")
    
    # Сохраняем
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✅ Отчёт сохранён: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute cluster metrics and classify roles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Семантические кластеры
  python compute_cluster_metrics.py \\
      --clusters_json results/full/stage2_umap_semantic_clusters.json \\
      --names_json results/full/stage2_cluster_names.json \\
      --graph_pkl results/full/stage2.1_graph_with_clusters.pkl \\
      --output_dir results/full/cluster_metrics/

  # Структурные кластеры (Louvain)
  python compute_cluster_metrics.py \\
      --clusters_json results/full/stage2_louvain_clusters.json \\
      --names_json results/full/stage2_louvain_names.json \\
      --graph_pkl results/full/stage2.1_graph_with_clusters.pkl \\
      --output_dir results/full/louvain_metrics/ \\
      --top_k 10 --min_weight 0.5
        """
    )
    
    base_dir = Path(__file__).parent
    
    parser.add_argument('--clusters_json', 
                       default=str(base_dir / 'stage2.1/results/step3_clusters/stage2_umap_semantic_clusters.json'),
                       help='JSON с кластерами (должен содержать clusters[].nearest_clusters)')
    parser.add_argument('--names_json', 
                       default=str(base_dir / 'stage2.1/results/step3_clusters/cluster_labels.json'),
                       help='JSON с названиями кластеров')
    parser.add_argument('--graph_pkl', 
                       default=str(base_dir / 'stage2.1/results/step2_models/graph_stage2_umap_k40.pkl'),
                       help='Pickle файл с NetworkX графом')
    parser.add_argument('--output_dir', 
                       default=str(base_dir / 'stage2.1/results/step4_analysis'),
                       help='Директория для сохранения результатов')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Сколько ближайших соседей учитывать (default: 5)')
    parser.add_argument('--min_weight', type=float, default=0.6,
                       help='Минимальный вес связи 0-1 (default: 0.6)')
    parser.add_argument('--embedding-key', type=str, default='embedding',
                       choices=['embedding', 'embedding_umap_150d'],
                       help='Ключ для embeddings: embedding (1536D) или embedding_umap_150d (150D UMAP)')
    
    args = parser.parse_args()
    
    # Создаём output директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем подпапки для новой структуры
    metrics_dir = output_dir / 'metrics'
    figures_dir = output_dir / 'figures'
    metrics_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("🎯 CLUSTER METRICS COMPUTATION")
    print("=" * 80)
    print(f"\nВходные данные:")
    print(f"  Clusters: {args.clusters_json}")
    print(f"  Names: {args.names_json}")
    print(f"  Graph: {args.graph_pkl}")
    print(f"  Output: {args.output_dir}")
    print(f"\nПараметры:")
    print(f"  Top-K neighbors: {args.top_k}")
    print(f"  Min weight: {args.min_weight}")
    print(f"  Embedding key: {args.embedding_key}")
    
    # 1. Загрузка данных
    clusters_data, names_data, G = load_data(
        args.clusters_json,
        args.names_json,
        args.graph_pkl
    )
    
    # 2. Построение графа кластеров
    CG = build_cluster_graph(
        clusters_data,
        names_data,
        top_k_neighbors=args.top_k,
        min_weight=args.min_weight,
        embedding_key=args.embedding_key
    )
    
    # 3. Вычисление метрик
    metrics = compute_graph_metrics(CG)
    
    # 4. Классификация ролей
    roles = classify_cluster_roles(CG, metrics)
    
    # 5. Поиск сообществ
    communities, node_to_community = detect_communities(CG)
    
    # 6. Визуализация
    plot_cluster_network_with_roles(CG, roles, figures_dir)
    plot_community_structure(CG, communities, roles, figures_dir)
    plot_importance_ranking(CG, roles, figures_dir)
    plot_opportunity_matrix(CG, roles, figures_dir)
    
    # 7. Сохранение результатов
    json_path = metrics_dir / 'cluster_metrics.json'  # Используем metrics_dir
    save_analysis_json(CG, roles, node_to_community, json_path)
    
    report_path = metrics_dir / 'cluster_report.txt'  # Используем metrics_dir
    save_text_report(CG, roles, report_path)
    
    print("\n" + "=" * 80)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 80)
    print(f"\n📂 Результаты:")
    print(f"  - JSON с метриками: {json_path}")
    print(f"  - Текстовый отчёт: {report_path}")
    print(f"\n💡 Метрики включают:")
    print(f"  - Graph centrality: degree, betweenness, closeness, PageRank, eigenvector")
    print(f"  - Cluster roles: roots, bridges, leaves, regular")
    print(f"  - Communities: {len(communities)} сообществ по Louvain")


if __name__ == '__main__':
    main()
