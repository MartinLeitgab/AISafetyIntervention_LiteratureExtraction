#!/usr/bin/env python3
"""
Извлечение представителей кластеров для LLM naming

Читает граф с кластерами и JSON с метаданными,
извлекает top-N представителей для каждого кластера
на основе близости к центроиду в исходном 1536D пространстве.
"""

import pickle
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


def load_graph_and_clusters(graph_path, clusters_json_path):
    """Загружает граф и метаданные кластеров"""
    print(f"📚 Загрузка графа: {graph_path}")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"📚 Загрузка clusters JSON: {clusters_json_path}")
    with open(clusters_json_path, 'r', encoding='utf-8') as f:
        clusters_data = json.load(f)
    
    print(f"  ✅ Граф: {G.number_of_nodes():,} нод")
    print(f"  ✅ Кластеров: {len(clusters_data['clusters'])}")
    
    return G, clusters_data


def extract_representatives_by_centroid(G, clusters_data, top_n=20):
    """
    Извлекает top-N представителей для каждого кластера
    на основе близости к центроиду в 1536D пространстве
    """
    print(f"\n🔍 Извлечение топ-{top_n} представителей для каждого кластера...")
    
    representatives = {}
    
    for cluster_info in tqdm(clusters_data['clusters'], desc="Обработка кластеров"):
        cluster_id = cluster_info['cluster_id']
        size = cluster_info['size']
        
        # Центроид в 150D (из UMAP) - не используем для ранжирования
        # centroid_150d = np.array(cluster_info['centroid_embedding'])
        
        # Собираем все ноды кластера с их 1536D embeddings
        cluster_nodes = []
        cluster_embeddings = []
        
        for node_id, node_data in G.nodes(data=True):
            if node_data.get('semantic_cluster') == cluster_id:
                if 'embedding' in node_data:
                    cluster_nodes.append({
                        'node_id': node_id,
                        'name': node_data.get('name', 'N/A'),
                        'description': node_data.get('description', '')
                    })
                    cluster_embeddings.append(node_data['embedding'])
        
        if not cluster_embeddings:
            print(f"  ⚠️  Кластер {cluster_id}: нет нод с embeddings")
            representatives[str(cluster_id)] = {
                'size': size,
                'representatives': []
            }
            continue
        
        # Вычисляем центроид в 1536D пространстве
        cluster_embeddings = np.array(cluster_embeddings)
        centroid_1536d = cluster_embeddings.mean(axis=0)
        
        # Вычисляем расстояния до центроида (cosine distance)
        distances = cosine_distances([centroid_1536d], cluster_embeddings)[0]
        
        # Сортируем по близости к центроиду
        sorted_indices = np.argsort(distances)
        
        # Берём топ-N
        top_representatives = []
        for idx in sorted_indices[:top_n]:
            node = cluster_nodes[idx]
            top_representatives.append(node['name'])
        
        representatives[str(cluster_id)] = {
            'size': size,
            'representatives': top_representatives
        }
        
        print(f"  Кластер {cluster_id}: {len(top_representatives)} представителей (размер: {size})")
    
    return representatives


def save_representatives(representatives, output_path):
    """Сохраняет представителей в JSON"""
    print(f"\n💾 Сохранение представителей: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(representatives, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Сохранено {len(representatives)} кластеров")


def main():
    parser = argparse.ArgumentParser(
        description='Извлечение представителей кластеров для LLM naming'
    )
    parser.add_argument('--graph_path', required=True, 
                       help='Путь к графу с кластерами (.pkl)')
    parser.add_argument('--clusters_json', required=True,
                       help='Путь к stage2_umap_semantic_clusters.json')
    parser.add_argument('--output', required=True,
                       help='Путь для сохранения представителей')
    parser.add_argument('--top_n', type=int, default=20,
                       help='Количество представителей на кластер (default: 20)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🎯 ИЗВЛЕЧЕНИЕ ПРЕДСТАВИТЕЛЕЙ КЛАСТЕРОВ")
    print("="*80)
    print(f"Граф: {args.graph_path}")
    print(f"Clusters JSON: {args.clusters_json}")
    print(f"Output: {args.output}")
    print(f"Top-N: {args.top_n}")
    
    # Загрузка
    G, clusters_data = load_graph_and_clusters(args.graph_path, args.clusters_json)
    
    # Извлечение представителей
    representatives = extract_representatives_by_centroid(G, clusters_data, top_n=args.top_n)
    
    # Сохранение
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_representatives(representatives, output_path)
    
    print("\n" + "="*80)
    print("✅ ЗАВЕРШЕНО")
    print("="*80)
    print(f"Результат сохранён: {output_path}")
    print(f"\nТеперь можно использовать для LLM naming:")
    print(f"  1. Открыть {output_path}")
    print(f"  2. Скопировать представителей каждого кластера")
    print(f"  3. Попросить LLM назвать кластер на основе представителей")


if __name__ == '__main__':
    main()
