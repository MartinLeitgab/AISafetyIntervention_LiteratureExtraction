"""
Stage 0: Collect Local Graphs

Загружает все локальные JSON графы из intervention_graph_creation/data/processed/
и объединяет их в единый NetworkX DiGraph без дедупликации.

Входные данные: JSON файлы {nodes: [...], edges: [...], meta: [...]}
                + embeddings/ папка с отдельными embedding файлами
Выходные данные: global_graph_stage0_raw.pkl
"""

import json
import pickle
from pathlib import Path
import networkx as nx
from typing import Dict, List, Any, Optional
from tqdm import tqdm


def load_embeddings_map(embeddings_dir: Path) -> Dict[str, List[float]]:
    """
    Загружает все embeddings из папки embeddings/ в словарь.
    
    Args:
        embeddings_dir: путь к папке с embedding файлами
        
    Returns:
        Словарь {id: embedding_vector}
    """
    embeddings = {}
    
    if not embeddings_dir.exists():
        return embeddings
    
    for emb_file in embeddings_dir.glob('*.json'):
        try:
            with open(emb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                emb_id = data.get('id')
                emb_vector = data.get('embedding')
                if emb_id and emb_vector:
                    embeddings[emb_id] = emb_vector
        except Exception as e:
            pass
    
    return embeddings


def load_local_graph(json_path: Path) -> Dict[str, Any]:
    """Загружает локальный граф из JSON файла."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_local_graph_to_global(
    G: nx.DiGraph, 
    local_graph: Dict[str, Any], 
    embeddings_map: Dict[str, List[float]],
    source_file: str,
    paper_dir: str
) -> None:
    """
    Добавляет локальный граф в глобальный без дедупликации.
    
    Args:
        G: глобальный граф
        local_graph: локальный граф {nodes: [...], edges: [...], meta: [...]}
        embeddings_map: словарь {id: embedding_vector} для присоединения embeddings
        source_file: имя файла источника (для отладки)
        paper_dir: имя папки со статьёй (для отладки)
    """
    nodes = local_graph.get('nodes', [])
    edges = local_graph.get('edges', [])
    meta = local_graph.get('meta', [])
    
    source_url = None
    source_value = None
    for m in meta:
        if m.get('key') == 'url':
            source_url = m.get('value')
        elif m.get('key') == 'source':
            source_value = m.get('value')
    
    for i, node in enumerate(nodes):
        node_id = f"{paper_dir}__node_{i}"
        node_name = node.get('name')
        
        if not node_name:
            continue
        
        attrs = {k: v for k, v in node.items()}
        attrs['source_file'] = source_file
        attrs['paper_dir'] = paper_dir
        if source_url:
            attrs['source_url'] = source_url
        if source_value:
            attrs['source'] = source_value
        
        node_hash = None
        for emb_id, emb_vec in embeddings_map.items():
            emb_data_path = Path(source_file).parent / 'embeddings' / f'{emb_id}.json'
            if emb_data_path.exists():
                with open(emb_data_path, 'r') as f:
                    emb_data = json.load(f)
                    if emb_data.get('type') == 'node' and node_name in emb_data.get('text', ''):
                        node_hash = emb_id
                        attrs['embedding'] = emb_vec
                        attrs['embedding_id'] = emb_id
                        break
        
        G.add_node(node_id, **attrs)
    
    for i, edge in enumerate(edges):
        source_node = edge.get('source_node')
        target_node = edge.get('target_node')
        
        if not source_node or not target_node:
            continue
        
        source_idx = None
        target_idx = None
        for j, node in enumerate(nodes):
            if node.get('name') == source_node:
                source_idx = j
            if node.get('name') == target_node:
                target_idx = j
        
        if source_idx is None or target_idx is None:
            continue
        
        source_id = f"{paper_dir}__node_{source_idx}"
        target_id = f"{paper_dir}__node_{target_idx}"
        
        attrs = {k: v for k, v in edge.items() if k not in ['source_node', 'target_node']}
        attrs['source_file'] = source_file
        attrs['paper_dir'] = paper_dir
        if source_url:
            attrs['source_url'] = source_url
        
        edge_desc = edge.get('description', '')
        for emb_id, emb_vec in embeddings_map.items():
            emb_data_path = Path(source_file).parent / 'embeddings' / f'{emb_id}.json'
            if emb_data_path.exists():
                with open(emb_data_path, 'r') as f:
                    emb_data = json.load(f)
                    if emb_data.get('type') == 'edge' and edge_desc in emb_data.get('text', ''):
                        attrs['embedding'] = emb_vec
                        attrs['embedding_id'] = emb_id
                        break
        
        G.add_edge(source_id, target_id, **attrs)


def collect_all_local_graphs(processed_dir: Path) -> nx.DiGraph:
    """
    Собирает все локальные графы в единый глобальный граф.
    
    Args:
        processed_dir: путь к директории с обработанными данными
        
    Returns:
        Глобальный NetworkX DiGraph со всеми локальными графами
    """
    G = nx.DiGraph()
    
    paper_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    print(f"Найдено {len(paper_dirs)} папок со статьями")
    
    errors = []
    skipped = 0
    
    for paper_dir in tqdm(paper_dirs, desc="Загрузка локальных графов"):
        paper_name = paper_dir.name
        main_json = paper_dir / f"{paper_name}.json"
        
        if not main_json.exists():
            skipped += 1
            continue
        
        try:
            embeddings_dir = paper_dir / 'embeddings'
            embeddings_map = load_embeddings_map(embeddings_dir)
            
            local_graph = load_local_graph(main_json)
            add_local_graph_to_global(G, local_graph, embeddings_map, str(main_json), paper_name)
        except Exception as e:
            errors.append((paper_name, str(e)))
    
    if skipped > 0:
        print(f"\n⚠️  Пропущено {skipped} папок без основного JSON файла")
    
    if errors:
        print(f"\n⚠️  Ошибки при загрузке {len(errors)} файлов:")
        for paper_name, error in errors[:10]:
            print(f"  {paper_name}: {error}")
        if len(errors) > 10:
            print(f"  ... и ещё {len(errors) - 10} ошибок")
    
    return G


def save_graph(G: nx.DiGraph, output_path: Path) -> None:
    """Сохраняет граф в pickle файл."""
    with open(output_path, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Граф сохранён: {output_path}")


def print_graph_stats(G: nx.DiGraph) -> None:
    """Выводит статистику по графу."""
    print("\n" + "="*60)
    print("СТАТИСТИКА STAGE 0 (RAW GRAPH)")
    print("="*60)
    print(f"Всего нод: {G.number_of_nodes()}")
    print(f"Всего рёбер: {G.number_of_edges()}")
    
    node_types = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nТипы нод:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")
    
    edge_types = {}
    for source, target, attrs in G.edges(data=True):
        edge_type = attrs.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("\nТипы рёбер:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"  {edge_type}: {count}")
    
    print("="*60)


def main():
    import sys
    
    base_dir = Path(__file__).parent.parent.parent
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        processed_dir = base_dir / "intervention_graph_creation" / "data" / "processed_test"
        print("🧪 ТЕСТОВЫЙ РЕЖИМ")
    else:
        processed_dir = base_dir / "intervention_graph_creation" / "data" / "processed"
    
    output_dir = base_dir / "local" / "graph_processing"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Stage 0: Сбор локальных графов")
    print(f"Источник: {processed_dir}")
    print(f"Выход: {output_dir / 'global_graph_stage0_raw.pkl'}")
    print()
    
    if not processed_dir.exists():
        print(f"❌ Директория не найдена: {processed_dir}")
        return
    
    G = collect_all_local_graphs(processed_dir)
    
    print_graph_stats(G)
    
    output_path = output_dir / "global_graph_stage0_raw.pkl"
    save_graph(G, output_path)
    
    print("\n✅ Stage 0 завершён успешно!")


if __name__ == "__main__":
    main()
