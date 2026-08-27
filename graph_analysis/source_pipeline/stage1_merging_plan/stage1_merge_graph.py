#!/usr/bin/env python3

import pickle
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Set, Tuple, List, Optional
from datetime import datetime
import argparse
import logging
import sys

import faiss


COSINE_SIMILARITY_THRESHOLD = 0.75
NAME_OVERLAP_THRESHOLD = 0.4
FAISS_N_CLUSTERS = 1000  # Количество кластеров для IVFFlat
FAISS_N_NEIGHBORS = 50  # Количество ближайших соседей для поиска

EDGE_TYPE_MAPPING_FILE = "edge_type_mapping.json"


def load_edge_type_mapping(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def name_overlap_jaccard(name1: str, aliases1: List[str], name2: str, aliases2: List[str]) -> float:
    set1 = {name1.lower()} | {a.lower() for a in aliases1}
    set2 = {name2.lower()} | {a.lower() for a in aliases2}
    
    intersection = set1 & set2
    union = set1 | set2
    
    if not union:
        return 0.0
    return len(intersection) / len(union)


def are_nodes_compatible(attrs1: Dict, attrs2: Dict) -> bool:
    if attrs1['type'] != attrs2['type']:
        return False
    
    if attrs1['type'] == 'concept':
        if attrs1.get('concept_category') != attrs2.get('concept_category'):
            return False
    
    return True


def merge_intervention_temporal_data(G: nx.DiGraph, canonical_id: str, duplicate_ids: List[str]) -> Dict:
    lifecycle_data = defaultdict(list)
    maturity_data = defaultdict(list)
    
    all_node_ids = [canonical_id] + duplicate_ids
    
    for node_id in all_node_ids:
        if node_id not in G.nodes:
            continue
        attrs = G.nodes[node_id]
        
        lc = attrs.get('intervention_lifecycle')
        mat = attrs.get('intervention_maturity')
        url = attrs.get('source_url', 'unknown')
        
        if lc is not None:
            lifecycle_data[int(lc)].append(url)
        if mat is not None:
            maturity_data[int(mat)].append(url)
    
    if not lifecycle_data or not maturity_data:
        return {
            'intervention_lifecycle': None,
            'intervention_maturity': None,
            'lifecycle_history': [],
            'maturity_history': []
        }
    
    main_lifecycle = max(lifecycle_data.keys(), key=lambda k: len(lifecycle_data[k]))
    main_maturity = max(maturity_data.keys(), key=lambda k: len(maturity_data[k]))
    
    lifecycle_history = [
        {"value": val, "source_urls": list(set(urls)), "count": len(urls)}
        for val, urls in sorted(lifecycle_data.items())
    ]
    
    maturity_history = [
        {"value": val, "source_urls": list(set(urls)), "count": len(urls)}
        for val, urls in sorted(maturity_data.items())
    ]
    
    return {
        'intervention_lifecycle': main_lifecycle,
        'intervention_maturity': main_maturity,
        'lifecycle_history': lifecycle_history,
        'maturity_history': maturity_history
    }


def choose_canonical_node(id1: str, attrs1: Dict, id2: str, attrs2: Dict) -> Tuple[str, Dict, str, Dict]:
    len1 = len(attrs1.get('description', ''))
    len2 = len(attrs2.get('description', ''))
    if len1 > len2:
        return (id1, attrs1, id2, attrs2)
    elif len2 > len1:
        return (id2, attrs2, id1, attrs1)
    
    aliases1 = len(attrs1.get('aliases', []))
    aliases2 = len(attrs2.get('aliases', []))
    if aliases1 > aliases2:
        return (id1, attrs1, id2, attrs2)
    elif aliases2 > aliases1:
        return (id2, attrs2, id1, attrs1)
    
    url1 = attrs1.get('source_url', '')
    url2 = attrs2.get('source_url', '')
    if url1 < url2:
        return (id1, attrs1, id2, attrs2)
    else:
        return (id2, attrs2, id1, attrs1)


def merge_node_attributes(G: nx.DiGraph, canonical_id: str, duplicate_ids: List[str]) -> Dict:
    canonical_attrs = G.nodes[canonical_id]
    merged = {}
    
    merged['name'] = canonical_attrs['name']
    merged['type'] = canonical_attrs['type']
    
    all_aliases = set(canonical_attrs.get('aliases', []))
    for dup_id in duplicate_ids:
        dup_attrs = G.nodes[dup_id]
        all_aliases.update(dup_attrs.get('aliases', []))
        all_aliases.add(dup_attrs['name'])
    all_aliases.discard(canonical_attrs['name'])
    merged['aliases'] = sorted(list(all_aliases))
    
    all_descriptions = [canonical_attrs.get('description', '')]
    for dup_id in duplicate_ids:
        all_descriptions.append(G.nodes[dup_id].get('description', ''))
    merged['description'] = max(all_descriptions, key=len, default='')
    
    all_rationales = [canonical_attrs.get('node_rationale', '')]
    for dup_id in duplicate_ids:
        all_rationales.append(G.nodes[dup_id].get('node_rationale', ''))
    merged['node_rationale'] = max(all_rationales, key=len, default='')
    
    merged['concept_category'] = canonical_attrs.get('concept_category')
    
    if canonical_attrs['type'] == 'intervention':
        temporal_data = merge_intervention_temporal_data(G, canonical_id, duplicate_ids)
        merged['intervention_lifecycle'] = temporal_data['intervention_lifecycle']
        merged['intervention_maturity'] = temporal_data['intervention_maturity']
        merged['lifecycle_history'] = temporal_data['lifecycle_history']
        merged['maturity_history'] = temporal_data['maturity_history']
        
        all_lifecycle_rationales = [canonical_attrs.get('intervention_lifecycle_rationale', '')]
        all_maturity_rationales = [canonical_attrs.get('intervention_maturity_rationale', '')]
        for dup_id in duplicate_ids:
            all_lifecycle_rationales.append(G.nodes[dup_id].get('intervention_lifecycle_rationale', ''))
            all_maturity_rationales.append(G.nodes[dup_id].get('intervention_maturity_rationale', ''))
        merged['intervention_lifecycle_rationale'] = max(all_lifecycle_rationales, key=len, default='')
        merged['intervention_maturity_rationale'] = max(all_maturity_rationales, key=len, default='')
    else:
        merged['intervention_lifecycle'] = None
        merged['intervention_maturity'] = None
        merged['lifecycle_history'] = None
        merged['maturity_history'] = None
        merged['intervention_lifecycle_rationale'] = None
        merged['intervention_maturity_rationale'] = None
    
    all_embeddings = []
    if canonical_attrs.get('embedding'):
        all_embeddings.append(np.array(canonical_attrs['embedding']))
    for dup_id in duplicate_ids:
        emb = G.nodes[dup_id].get('embedding')
        if emb:
            all_embeddings.append(np.array(emb))
    
    if all_embeddings:
        merged['embedding'] = np.mean(all_embeddings, axis=0).tolist()
    else:
        merged['embedding'] = None
    
    merged['source_url'] = canonical_attrs.get('source_url')
    merged['merge_count'] = len(duplicate_ids) + 1
    
    return merged


def find_merge_candidates_semantic(G: nx.DiGraph, logger: logging.Logger) -> Tuple[Set[Tuple[str, str]], List[Dict]]:
    print("\n=== Семантическое сходство (только embeddings) ===")
    print("Используется FAISS IndexIVFFlat для ускорения поиска")
    logger.info(f"\n{'='*60}")
    logger.info(f"Параметры поиска:")
    logger.info(f"  COSINE_SIMILARITY_THRESHOLD = {COSINE_SIMILARITY_THRESHOLD}")
    logger.info(f"  NAME_OVERLAP_THRESHOLD = {NAME_OVERLAP_THRESHOLD}")
    logger.info(f"{'='*60}\n")
    
    candidates = set()
    all_candidates_data = []  # Собираем ВСЕ кандидаты с метриками
    
    blocks = defaultdict(list)
    for node_id, attrs in G.nodes(data=True):
        if attrs['type'] == 'concept':
            block_key = (attrs['type'], attrs.get('concept_category'))
        else:
            # Interventions: block only by type to enable temporal evolution merging
            block_key = (attrs['type'],)
        blocks[block_key].append(node_id)
    
    for block_key, node_ids in tqdm(blocks.items(), desc="Обработка блоков"):
        if len(node_ids) < 2:
            continue
        
        embeddings = []
        valid_node_ids = []
        
        for node_id in node_ids:
            emb = G.nodes[node_id].get('embedding')
            if emb is not None:
                embeddings.append(emb)
                valid_node_ids.append(node_id)
        
        if len(embeddings) < 2:
            continue
        
        embeddings_matrix = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_matrix)
        
        dimension = embeddings_matrix.shape[1]
        n_nodes = len(embeddings_matrix)
        
        n_clusters = min(FAISS_N_CLUSTERS, n_nodes // 10) if n_nodes > 100 else 1
        
        if n_clusters > 1:
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings_matrix)
            index.add(embeddings_matrix)
            index.nprobe = min(10, n_clusters)
        else:
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_matrix)
        
        k = min(FAISS_N_NEIGHBORS, n_nodes)
        distances, indices = index.search(embeddings_matrix, k)
        
        for i in range(len(valid_node_ids)):
            node1_id = valid_node_ids[i]
            node1_attrs = G.nodes[node1_id]
            
            for j, similarity in enumerate(distances[i]):
                neighbor_idx = indices[i][j]
                
                if neighbor_idx == i:
                    continue
                
                if similarity < COSINE_SIMILARITY_THRESHOLD:
                    continue
                
                node2_id = valid_node_ids[neighbor_idx]
                
                if node1_id >= node2_id:
                    continue
                
                node2_attrs = G.nodes[node2_id]
                
                name_overlap = name_overlap_jaccard(
                    node1_attrs['name'], node1_attrs.get('aliases', []),
                    node2_attrs['name'], node2_attrs.get('aliases', [])
                )
                
                if name_overlap >= NAME_OVERLAP_THRESHOLD:
                    candidates.add((node1_id, node2_id))
                    
                    # Сохраняем ВСЕ кандидаты с их метриками
                    all_candidates_data.append({
                        'node1_id': node1_id,
                        'node1_name': node1_attrs['name'],
                        'node1_aliases': node1_attrs.get('aliases', []),
                        'node2_id': node2_id,
                        'node2_name': node2_attrs['name'],
                        'node2_aliases': node2_attrs.get('aliases', []),
                        'cosine_similarity': float(similarity),
                        'name_overlap': float(name_overlap),
                        'type': node1_attrs['type'],
                        'category': node1_attrs.get('concept_category') or node1_attrs.get('intervention_lifecycle')
                    })
    
    print(f"Найдено кандидатов: {len(candidates)}")
    
    # Логируем примеры с МИНИМАЛЬНЫМИ значениями (нижняя граница)
    if all_candidates_data:
        # Сортируем по сумме метрик (чтобы найти самые "граничные" случаи)
        sorted_by_metrics = sorted(all_candidates_data, 
                                   key=lambda x: (x['cosine_similarity'] + x['name_overlap']))
        
        # Берем первые 10 (с минимальными метриками)
        bottom_examples = sorted_by_metrics[:10]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ПРИМЕРЫ С МИНИМАЛЬНЫМИ МЕТРИКАМИ (нижняя граница, {len(bottom_examples)} шт.):")
        logger.info(f"{'='*60}\n")
        
        for idx, ex in enumerate(bottom_examples, 1):
            logger.info(f"Пример {idx}:")
            logger.info(f"  Тип: {ex['type']}, Категория: {ex['category']}")
            logger.info(f"  Node 1: {ex['node1_name']}")
            logger.info(f"    ID: {ex['node1_id']}")
            logger.info(f"    Aliases: {ex['node1_aliases']}")
            logger.info(f"  Node 2: {ex['node2_name']}")
            logger.info(f"    ID: {ex['node2_id']}")
            logger.info(f"    Aliases: {ex['node2_aliases']}")
            logger.info(f"  📊 Cosine Similarity: {ex['cosine_similarity']:.4f}")
            logger.info(f"  📊 Name Overlap (Jaccard): {ex['name_overlap']:.4f}")
            logger.info("")
        
        # Добавим статистику по метрикам
        cosines = [x['cosine_similarity'] for x in all_candidates_data]
        overlaps = [x['name_overlap'] for x in all_candidates_data]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"СТАТИСТИКА ПО МЕТРИКАМ КАНДИДАТОВ:")
        logger.info(f"{'='*60}")
        logger.info(f"Cosine Similarity:")
        logger.info(f"  Min: {min(cosines):.4f}")
        logger.info(f"  Max: {max(cosines):.4f}")
        logger.info(f"  Mean: {np.mean(cosines):.4f}")
        logger.info(f"  Median: {np.median(cosines):.4f}")
        logger.info(f"\nName Overlap (Jaccard):")
        logger.info(f"  Min: {min(overlaps):.4f}")
        logger.info(f"  Max: {max(overlaps):.4f}")
        logger.info(f"  Mean: {np.mean(overlaps):.4f}")
        logger.info(f"  Median: {np.median(overlaps):.4f}")
        logger.info("")
    
    return candidates, all_candidates_data


def apply_node_merging(G: nx.DiGraph, merge_candidates: Set[Tuple[str, str]]) -> Dict[str, str]:
    print("\n=== Применение мержинга нод ===")
    
    parent = {}
    
    def find(node_id):
        if node_id not in parent:
            parent[node_id] = node_id
        if parent[node_id] != node_id:
            parent[node_id] = find(parent[node_id])
        return parent[node_id]
    
    def union(node1_id, node2_id):
        root1 = find(node1_id)
        root2 = find(node2_id)
        if root1 != root2:
            parent[root2] = root1
    
    for node1_id, node2_id in tqdm(merge_candidates, desc="Группировка нод"):
        union(node1_id, node2_id)
    
    groups = defaultdict(list)
    for node_id in G.nodes():
        canonical_id = find(node_id)
        if node_id != canonical_id:
            groups[canonical_id].append(node_id)
    
    node_mapping = {}
    
    for canonical_id, duplicate_ids in tqdm(groups.items(), desc="Выбор canonical нод"):
        all_nodes = [canonical_id] + duplicate_ids
        
        current_canonical = canonical_id
        current_attrs = G.nodes[current_canonical]
        
        for node_id in duplicate_ids:
            node_attrs = G.nodes[node_id]
            current_canonical, current_attrs, _, _ = choose_canonical_node(
                current_canonical, current_attrs, node_id, node_attrs
            )
        
        for node_id in all_nodes:
            if node_id != current_canonical:
                node_mapping[node_id] = current_canonical
    
    for canonical_id, duplicate_ids in tqdm(groups.items(), desc="Объединение атрибутов"):
        actual_canonical = node_mapping.get(canonical_id, canonical_id)
        
        all_nodes = [canonical_id] + duplicate_ids
        total_merged = len(all_nodes)
        
        canonical_attrs = G.nodes[actual_canonical]
        all_names = {canonical_attrs['name']}
        all_aliases = set(canonical_attrs.get('aliases', []))
        all_descriptions = [canonical_attrs.get('description', '')]
        all_embeddings = [canonical_attrs.get('embedding')]
        
        for node_id in all_nodes:
            if node_id == actual_canonical:
                continue
            if node_id not in G.nodes:
                continue
            dup_attrs = G.nodes[node_id]
            all_names.add(dup_attrs['name'])
            all_aliases.update(dup_attrs.get('aliases', []))
            all_descriptions.append(dup_attrs.get('description', ''))
            all_embeddings.append(dup_attrs.get('embedding'))
        
        canonical_attrs['aliases'] = list(all_names | all_aliases)
        canonical_attrs['description'] = max(all_descriptions, key=len) if all_descriptions else ''
        
        valid_embeddings = [e for e in all_embeddings if e is not None]
        if valid_embeddings:
            canonical_attrs['embedding'] = np.mean(valid_embeddings, axis=0).tolist()
        
        canonical_attrs['merge_count'] = total_merged
    
    nodes_to_remove = set(node_mapping.keys())
    G.remove_nodes_from(nodes_to_remove)
    
    for node_id, attrs in G.nodes(data=True):
        if 'merge_count' not in attrs:
            attrs['merge_count'] = 1
    
    print(f"Удалено нод: {len(nodes_to_remove)}")
    return node_mapping


def normalize_edge_type(edge_type: str, mapping: Dict) -> Tuple[Optional[str], bool]:
    if edge_type in mapping['delete']:
        return None, False
    
    if edge_type in mapping['simple_rename']:
        return mapping['simple_rename'][edge_type], False
    
    if edge_type in mapping['rename_with_reverse']:
        return mapping['rename_with_reverse'][edge_type], True
    
    return edge_type, False


def update_and_normalize_edges(G: nx.DiGraph, node_mapping: Dict[str, str], edge_mapping: Dict) -> int:
    print("\n=== Обновление и нормализация рёбер ===")
    
    new_edges = []
    self_loops_removed = 0
    
    for source, target, attrs in tqdm(list(G.edges(data=True)), desc="Обработка рёбер"):
        new_source = node_mapping.get(source, source)
        new_target = node_mapping.get(target, target)
        
        if new_source == new_target:
            self_loops_removed += 1
            continue
        
        edge_type = attrs.get('type')
        new_type, should_reverse = normalize_edge_type(edge_type, edge_mapping)
        
        if new_type is None:
            continue
        
        attrs['type'] = new_type
        
        if should_reverse:
            new_edges.append((new_target, new_source, attrs))
        else:
            new_edges.append((new_source, new_target, attrs))
    
    G.clear_edges()
    
    for source, target, attrs in new_edges:
        G.add_edge(source, target, **attrs)
    
    for _, _, attrs in G.edges(data=True):
        if 'merge_count' not in attrs:
            attrs['merge_count'] = 1
    
    print(f"Self-loops удалено: {self_loops_removed}")
    return self_loops_removed


def merge_duplicate_edges(G: nx.DiGraph) -> Dict:
    print("\n=== Мержинг дубликатов рёбер ===")
    
    stats = {
        'total_edges_before': G.number_of_edges(),
        'edges_merged': 0,
        'total_edges_after': 0
    }
    
    edge_groups = defaultdict(list)
    
    for source, target, attrs in tqdm(list(G.edges(data=True)), desc="Группировка рёбер"):
        edge_type = attrs.get('type')
        edge_conf = attrs.get('edge_confidence', 1)
        group_key = (source, edge_type, target, edge_conf)
        edge_groups[group_key].append(attrs)
    
    G.clear_edges()
    
    for (source, edge_type, target, edge_conf), edges_list in tqdm(edge_groups.items(), desc="Объединение рёбер"):
        if len(edges_list) == 1:
            G.add_edge(source, target, **edges_list[0])
        else:
            merged_attrs = merge_edge_attributes(edges_list)
            G.add_edge(source, target, **merged_attrs)
            stats['edges_merged'] += len(edges_list) - 1
    
    stats['total_edges_after'] = G.number_of_edges()
    
    print(f"Рёбер объединено: {stats['edges_merged']}")
    return stats


def merge_edge_attributes(edge_attrs_list: List[Dict]) -> Dict:
    if not edge_attrs_list:
        return {}
    
    merged = {}
    
    merged['type'] = edge_attrs_list[0]['type']
    
    all_descriptions = [e.get('description', '') for e in edge_attrs_list]
    merged['description'] = max(all_descriptions, key=len, default='')
    
    all_confidences = [e.get('edge_confidence', 1) for e in edge_attrs_list]
    merged['edge_confidence'] = max(all_confidences)
    
    all_conf_rationales = [e.get('edge_confidence_rationale', '') for e in edge_attrs_list]
    merged['edge_confidence_rationale'] = max(all_conf_rationales, key=len, default='')
    
    all_rationales = [e.get('edge_rationale', '') for e in edge_attrs_list]
    merged['edge_rationale'] = max(all_rationales, key=len, default='')
    
    all_embeddings = []
    for e in edge_attrs_list:
        emb = e.get('embedding')
        if emb:
            all_embeddings.append(np.array(emb))
    
    if all_embeddings:
        merged['embedding'] = np.mean(all_embeddings, axis=0).tolist()
    else:
        merged['embedding'] = None
    
    merged['source_url'] = edge_attrs_list[0].get('source_url')
    merged['merge_count'] = len(edge_attrs_list)
    
    return merged


def collect_statistics(G: nx.DiGraph, stats: Dict) -> Dict:
    print("\n=== Сбор статистики ===")
    
    edge_types = defaultdict(int)
    for _, _, attrs in G.edges(data=True):
        edge_types[attrs.get('type')] += 1
    
    node_merge_counts = defaultdict(int)
    for _, attrs in G.nodes(data=True):
        count = attrs.get('merge_count', 1)
        node_merge_counts[count] += 1
    
    edge_merge_counts = defaultdict(int)
    for _, _, attrs in G.edges(data=True):
        count = attrs.get('merge_count', 1)
        edge_merge_counts[count] += 1
    
    stats['nodes_after'] = G.number_of_nodes()
    stats['edges_after'] = G.number_of_edges()
    stats['edge_types'] = dict(sorted(edge_types.items(), key=lambda x: x[1], reverse=True))
    stats['node_merge_distribution'] = dict(sorted(node_merge_counts.items()))
    stats['edge_merge_distribution'] = dict(sorted(edge_merge_counts.items()))
    
    return stats


def print_statistics(stats: Dict, logger: logging.Logger):
    logger.info("\n" + "="*60)
    logger.info("ИТОГОВАЯ СТАТИСТИКА")
    logger.info("="*60)
    
    logger.info(f"\nНОДЫ:")
    logger.info(f"  До мержинга:  {stats['nodes_before']:,}")
    logger.info(f"  После мержинга: {stats['nodes_after']:,}")
    logger.info(f"  Удалено: {stats['nodes_before'] - stats['nodes_after']:,} ({100*(stats['nodes_before'] - stats['nodes_after'])/stats['nodes_before']:.1f}%)")
    
    logger.info(f"\nРЁБРА:")
    logger.info(f"  До мержинга:  {stats['edges_before']:,}")
    logger.info(f"  Self-loops удалено: {stats.get('self_loops_removed', 0):,}")
    logger.info(f"  Рёбер объединено: {stats.get('edges_merged', 0):,}")
    logger.info(f"  После мержинга: {stats['edges_after']:,}")
    logger.info(f"  Удалено: {stats['edges_before'] - stats['edges_after']:,} ({100*(stats['edges_before'] - stats['edges_after'])/stats['edges_before']:.1f}%)")
    
    logger.info(f"\nТИПЫ РЁБЕР ({len(stats['edge_types'])}):")  
    for edge_type, count in stats['edge_types'].items():
        logger.info(f"  {edge_type}: {count:,}")
    
    logger.info(f"\nРАСПРЕДЕЛЕНИЕ MERGE_COUNT (ноды):")
    for count, num_nodes in stats['node_merge_distribution'].items():
        logger.info(f"  {count} нод(ы) объединено: {num_nodes:,} групп")
    
    logger.info(f"\nРАСПРЕДЕЛЕНИЕ MERGE_COUNT (рёбра):")
    for count, num_edges in stats['edge_merge_distribution'].items():
        logger.info(f"  {count} рёбер объединено: {num_edges:,} групп")
def setup_logging(log_file: Path):
    """Настройка логирования в файл и консоль."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Мержинг и дедупликация графа',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  python3 stage1_merge_graph.py --graph_path ../stage0_graph_processing/global_graph_stage0_raw.pkl
  python3 stage1_merge_graph.py --graph_path test_20k.pkl --output stage1_test_merged.pkl
        '''
    )
    parser.add_argument(
        '--graph_path',
        type=str,
        default='../stage0_graph_processing/global_graph_stage0_raw.pkl',
        help='Путь к входному графу (pickle файл)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='global_graph_stage1_merged.pkl',
        help='Путь к выходному графу (pickle файл)'
    )
    parser.add_argument(
        '--log',
        type=str,
        default=None,
        help='Путь к лог файлу (по умолчанию: stage1_run_<timestamp>.log)'
    )
    parser.add_argument(
        '--cosine_threshold',
        type=float,
        default=0.75,
        help='Порог cosine similarity (по умолчанию: 0.75)'
    )
    parser.add_argument(
        '--name_overlap_threshold',
        type=float,
        default=0.4,
        help='Порог name overlap (по умолчанию: 0.4)'
    )
    
    args = parser.parse_args()
    
    global COSINE_SIMILARITY_THRESHOLD, NAME_OVERLAP_THRESHOLD
    COSINE_SIMILARITY_THRESHOLD = args.cosine_threshold
    NAME_OVERLAP_THRESHOLD = args.name_overlap_threshold
    
    input_graph = Path(args.graph_path)
    output_graph = Path(args.output)
    
    if args.log:
        log_file = Path(args.log)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(f'stage1_run_{timestamp}.log')
    
    logger = setup_logging(log_file)
    
    logger.info("="*60)
    logger.info("STAGE 1: Мержинг и дедупликация графа")
    logger.info("="*60)
    logger.info(f"Лог файл: {log_file}")
    logger.info(f"Входной граф: {input_graph}")
    logger.info(f"Выходной граф: {output_graph}")
    
    if not input_graph.exists():
        logger.error(f"❌ Файл не найден: {input_graph}")
        return
    
    logger.info(f"\nЗагрузка графа из {input_graph}...")
    with open(input_graph, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"Загружено: {G.number_of_nodes():,} нод, {G.number_of_edges():,} рёбер")
    
    stats = {
        'nodes_before': G.number_of_nodes(),
        'edges_before': G.number_of_edges()
    }
    
    edge_mapping = load_edge_type_mapping(EDGE_TYPE_MAPPING_FILE)
    
    all_candidates, examples = find_merge_candidates_semantic(G, logger)
    logger.info(f"\nВсего уникальных кандидатов на мержинг: {len(all_candidates)}")
    
    node_mapping = apply_node_merging(G, all_candidates)
    
    self_loops_removed = update_and_normalize_edges(G, node_mapping, edge_mapping)
    stats['self_loops_removed'] = self_loops_removed
    
    edge_stats = merge_duplicate_edges(G)
    stats.update(edge_stats)
    
    stats = collect_statistics(G, stats)
    
    logger.info(f"\nСохранение графа в {output_graph}...")
    with open(output_graph, 'wb') as f:
        pickle.dump(G, f, protocol=5)
    
    print_statistics(stats, logger)
    
    logger.info("\n✅ ЗАВЕРШЕНО")
    logger.info(f"Результаты сохранены:")
    logger.info(f"  - Граф: {output_graph}")
    logger.info(f"  - Лог: {log_file}")


if __name__ == "__main__":
    main()
