# metrics_analyzer.py
import networkx as nx
import numpy as np
from collections import Counter
from datetime import datetime
import json
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
from models.models import CausalNode, CausalEdge
from utilities.utils import logger, MATPLOTLIB_AVAILABLE
from typing import Dict, List, Set, Tuple, Optional, Union
import matplotlib.pyplot as plt
class MetricsAnalyzer:
    """Handles comprehensive graph analysis and metric computation"""
    
    def __init__(self, graph: nx.DiGraph, nodes: Dict[str, CausalNode], edges: Dict[str, List[CausalEdge]]):
        self.graph = graph
        self.nodes = nodes
        self.edges = edges

    def analyze_graph_metrics(self, export_filename: str = None, show_plots: bool = True):
        """Comprehensive graph analysis with multiple metrics and visualizations"""
        if not self.nodes:
            print("No nodes in graph to analyze")
            return {}

        print("\n" + "="*80)
        print("COMPREHENSIVE GRAPH ANALYSIS")
        print("="*80)

        metrics = self._calculate_comprehensive_metrics()
        self._print_analysis_summary(metrics)
        if MATPLOTLIB_AVAILABLE and show_plots:
            self._create_analysis_visualizations(metrics)
        if export_filename:
            self._export_analysis_metrics(metrics, export_filename)

        return metrics

    def _calculate_comprehensive_metrics(self):
        """Calculate all graph metrics"""
        metrics = {}

        # Basic graph properties
        metrics['basic'] = {
            'total_nodes': len(self.nodes),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'num_weakly_connected_components': nx.number_weakly_connected_components(self.graph),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.graph)
        }

        # Node degrees
        degrees = dict(self.graph.degree())
        metrics['degrees'] = {
            'node_degrees': degrees,
            'degree_sequence': list(degrees.values()),
            'avg_degree': np.mean(list(degrees.values())) if degrees else 0,
            'max_degree': max(degrees.values()) if degrees else 0,
            'min_degree': min(degrees.values()) if degrees else 0
        }

        # In/out degree analysis
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        metrics['directed_degrees'] = {
            'in_degrees': in_degrees,
            'out_degrees': out_degrees,
            'avg_in_degree': np.mean(list(in_degrees.values())) if in_degrees else 0,
            'avg_out_degree': np.mean(list(out_degrees.values())) if out_degrees else 0
        }

        # Text occurrence analysis
        node_texts = [node.concept_text for node in self.nodes.values()]
        edge_texts = [edge.edge_text for edges in self.edges.values() for edge in edges]

        metrics['text_analysis'] = {
            'node_text_counts': Counter(node_texts),
            'edge_text_counts': Counter(edge_texts),
            'unique_node_concepts': len(set(node_texts)),
            'unique_edge_relationships': len(set(edge_texts))
        }

        # Intervention analysis
        interventions = [node for node in self.nodes.values() if node.isIntervention == 1]
        problems = [node for node in self.nodes.values() if node.isIntervention == 0]

        metrics['interventions'] = {
            'total_interventions': len(interventions),
            'total_problems': len(problems),
            'intervention_ratio': len(interventions) / len(self.nodes) if self.nodes else 0,
            'implemented_interventions': len([i for i in interventions if i.implemented == 1]),
            'mature_interventions': len([i for i in interventions if i.maturity_level and i.maturity_level >= 3])
        }

        # Connected components analysis
        if self.graph.nodes():
            weakly_connected = list(nx.weakly_connected_components(self.graph))
            strongly_connected = list(nx.strongly_connected_components(self.graph))

            metrics['components'] = {
                'weakly_connected_components': weakly_connected,
                'strongly_connected_components': strongly_connected,
                'largest_wcc_size': len(max(weakly_connected, key=len)) if weakly_connected else 0,
                'largest_scc_size': len(max(strongly_connected, key=len)) if strongly_connected else 0,
                'wcc_sizes': [len(comp) for comp in weakly_connected],
                'scc_sizes': [len(comp) for comp in strongly_connected]
            }
        else:
            metrics['components'] = {
                'weakly_connected_components': [],
                'strongly_connected_components': [],
                'largest_wcc_size': 0,
                'largest_scc_size': 0,
                'wcc_sizes': [],
                'scc_sizes': []
            }

        # Centrality measures
        if self.graph.nodes():
            try:
                betweenness = nx.betweenness_centrality(self.graph)
                closeness = nx.closeness_centrality(self.graph)
                pagerank = nx.pagerank(self.graph)

                metrics['centrality'] = {
                    'betweenness_centrality': betweenness,
                    'closeness_centrality': closeness,
                    'pagerank': pagerank,
                    'avg_betweenness': np.mean(list(betweenness.values())),
                    'avg_closeness': np.mean(list(closeness.values())),
                    'avg_pagerank': np.mean(list(pagerank.values()))
                }
            except:
                metrics['centrality'] = {'error': 'Could not calculate centrality measures'}

        # Community detection
        if self.graph.nodes() and len(self.graph.nodes()) > 1:
            try:
                # Convert to undirected for community detection
                undirected_graph = self.graph.to_undirected()
                communities = community_louvain.best_partition(undirected_graph)
                modularity = community_louvain.modularity(communities, undirected_graph)

                metrics['communities'] = {
                    'node_communities': communities,
                    'modularity': modularity,
                    'num_communities': len(set(communities.values())),
                    'community_sizes': Counter(communities.values())
                }
            except Exception as e:
                metrics['communities'] = {'error': f'Could not detect communities: {str(e)}'}

        # Clustering coefficient
        if self.graph.nodes():
            try:
                clustering = nx.clustering(self.graph.to_undirected())
                metrics['clustering'] = {
                    'node_clustering': clustering,
                    'avg_clustering': nx.average_clustering(self.graph.to_undirected()),
                    'transitivity': nx.transitivity(self.graph.to_undirected())
                }
            except:
                metrics['clustering'] = {'error': 'Could not calculate clustering'}

        return metrics
    
    def _print_analysis_summary(self, metrics):
        """Print comprehensive analysis summary"""
        basic = metrics['basic']
        degrees = metrics['degrees']
        interventions = metrics['interventions']
        components = metrics['components']

        print(f"Basic Graph Properties:")
        print(f"  • Total Nodes: {basic['total_nodes']:,}")
        print(f"  • Total Edges: {basic['total_edges']:,}")
        print(f"  • Graph Density: {basic['density']:.4f}")
        print(f"  • Weakly Connected: {basic['is_connected']}")
        print(f"  • Connected Components (Weak): {basic['num_weakly_connected_components']}")
        print(f"  • Connected Components (Strong): {basic['num_strongly_connected_components']}")

        print(f"\nDegree Analysis:")
        print(f"  • Average Degree: {degrees['avg_degree']:.2f}")
        print(f"  • Maximum Degree: {degrees['max_degree']}")
        print(f"  • Minimum Degree: {degrees['min_degree']}")

        print(f"\nIntervention Analysis:")
        print(f"  • Total Interventions: {interventions['total_interventions']}")
        print(f"  • Total Problems/Concepts: {interventions['total_problems']}")
        print(f"  • Intervention Ratio: {interventions['intervention_ratio']:.2f}")
        print(f"  • Implemented Interventions: {interventions['implemented_interventions']}")
        print(f"  • Mature Interventions: {interventions['mature_interventions']}")

        print(f"\nConnected Components:")
        print(f"  • Largest WCC Size: {components['largest_wcc_size']}")
        print(f"  • Largest SCC Size: {components['largest_scc_size']}")

        # Text analysis highlights
        text_analysis = metrics['text_analysis']
        most_common_nodes = text_analysis['node_text_counts'].most_common(3)
        least_common_nodes = [item for item in text_analysis['node_text_counts'].items() if item[1] == 1][:3]

        print(f"\nText Occurrence Analysis:")
        print(f"  • Most Common Node Concepts:")
        for concept, count in most_common_nodes:
            print(f"    - '{concept[:50]}...' ({count} occurrences)")
        print(f"  • Unique Concepts (1 occurrence): {len(least_common_nodes)}")

        # Centrality highlights
        if 'centrality' in metrics and 'error' not in metrics['centrality']:
            centrality = metrics['centrality']
            print(f"\nCentrality Measures:")
            print(f"  • Average Betweenness Centrality: {centrality['avg_betweenness']:.4f}")
            print(f"  • Average Closeness Centrality: {centrality['avg_closeness']:.4f}")
            print(f"  • Average PageRank: {centrality['avg_pagerank']:.4f}")

        # Community analysis
        if 'communities' in metrics and 'error' not in metrics['communities']:
            communities = metrics['communities']
            print(f"\nCommunity Structure:")
            print(f"  • Number of Communities: {communities['num_communities']}")
            print(f"  • Modularity Score: {communities['modularity']:.4f}")

    def _create_analysis_visualizations(self, metrics):
        # ----- PAGE 1: Structural Metrics -----
        fig1 = plt.figure(figsize=(18, 18))

        # Degree Distribution
        plt.subplot(3, 3, 1)
        degrees = metrics['degrees']['degree_sequence']
        plt.hist(degrees, bins=min(20, len(set(degrees))), alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Node Degree')
        plt.ylabel('Frequency')
        plt.title('Node Degree Distribution')
        plt.grid(True, alpha=0.3)

        # Node Index vs Degree
        plt.subplot(3, 3, 2)
        plt.scatter(range(len(degrees)), degrees, alpha=0.6, color='coral')
        plt.xlabel('Node Index')
        plt.ylabel('Degree')
        plt.title('Node Index vs Degree')
        plt.grid(True, alpha=0.3)

        # In/Out Degree
        plt.subplot(3, 3, 3)
        in_degrees = list(metrics['directed_degrees']['in_degrees'].values())
        out_degrees = list(metrics['directed_degrees']['out_degrees'].values())
        plt.hist(in_degrees, bins=10, alpha=0.7, label='In-degree', color='lightgreen')
        plt.hist(out_degrees, bins=10, alpha=0.7, label='Out-degree', color='lightcoral')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('In/Out Degree Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Connected Component Size
        plt.subplot(3, 3, 4)
        wcc_sizes = metrics['components']['wcc_sizes']
        plt.hist(wcc_sizes, bins=min(15, len(set(wcc_sizes))), alpha=0.7, color='gold', edgecolor='black')
        plt.xlabel('Component Size')
        plt.ylabel('Frequency')
        plt.title('Connected Component Size Distribution')
        plt.grid(True, alpha=0.3)

        # Largest Connected Components
        plt.subplot(3, 3, 5)
        largest_components = sorted(wcc_sizes, reverse=True)[:10]
        plt.bar(range(len(largest_components)), largest_components, color='mediumpurple')
        plt.xlabel('Component Rank')
        plt.ylabel('Number of Nodes')
        plt.title('10 Largest Connected Components')
        plt.grid(True, alpha=0.3)

        # Nodes vs Edges Correlation
        plt.subplot(3, 3, 6)
        component_edges = [self.graph.subgraph(comp).number_of_edges() for comp in metrics['components']['weakly_connected_components']]
        plt.scatter(wcc_sizes, component_edges, alpha=0.7, color='darkseagreen')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Number of Edges')
        plt.title('Nodes vs Edges Correlation')
        plt.grid(True, alpha=0.3)
        if len(wcc_sizes) > 1:
            corr = np.corrcoef(wcc_sizes, component_edges)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

        # Centrality
        plt.subplot(3, 3, 7)
        betweenness = list(metrics['centrality']['betweenness_centrality'].values())
        plt.hist(betweenness, bins=15, alpha=0.7, color='orchid', edgecolor='black')
        plt.xlabel('Betweenness Centrality')
        plt.ylabel('Frequency')
        plt.title('Betweenness Centrality Distribution')
        plt.grid(True, alpha=0.3)

        # Clustering Coefficient
        plt.subplot(3, 3, 8)
        clustering = list(metrics['clustering']['node_clustering'].values())
        plt.hist(clustering, bins=15, alpha=0.7, color='sandybrown', edgecolor='black')
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Frequency')
        plt.title('Node Clustering Coefficient Distribution')
        plt.grid(True, alpha=0.3)

        # Summary Stats
        plt.subplot(3, 3, 9)
        plt.axis('off')
        summary_text = f"""
        GRAPH SUMMARY STATISTICS

        Total Nodes: {metrics['basic']['total_nodes']:,}
        Total Edges: {metrics['basic']['total_edges']:,}
        Density: {metrics['basic']['density']:.4f}

        Avg Degree: {metrics['degrees']['avg_degree']:.2f}
        Max Degree: {metrics['degrees']['max_degree']}

        Interventions: {metrics['interventions']['total_interventions']}
        Problems: {metrics['interventions']['total_problems']}

        Largest Component: {metrics['components']['largest_wcc_size']} nodes
        Num Components: {metrics['basic']['num_weakly_connected_components']}
        """
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)
        fig1.suptitle('Causal Graph Analysis — Structure', fontsize=16)
        plt.show()
        
        # ----- PAGE 2: Concept and Intervention Analysis -----
        fig2 = plt.figure(figsize=(18, 12))

        # Top Node Concepts
        plt.subplot(2, 2, 1)
        top_concepts = dict(metrics['text_analysis']['node_text_counts'].most_common(10))
        names = [name[:20] + '...' if len(name) > 20 else name for name in top_concepts.keys()]
        plt.barh(range(len(names)), list(top_concepts.values()), color='lightblue')
        plt.yticks(range(len(names)), names)
        plt.xlabel('Occurrence Count')
        plt.title('Top 10 Node Concepts by Occurrence')
        plt.gca().invert_yaxis()

        # Intervention vs Problem Pie Chart
        plt.subplot(2, 2, 2)
        labels = ['Problems/Concepts', 'Interventions']
        sizes = [metrics['interventions']['total_problems'], metrics['interventions']['total_interventions']]
        colors = ['lightcoral', 'lightgreen']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Intervention vs Problem Distribution')

        # Community Sizes
        if 'communities' in metrics and 'error' not in metrics['communities']:
            plt.subplot(2, 2, 3)
            sizes = list(metrics['communities']['community_sizes'].values())
            plt.hist(sizes, bins=min(10, len(set(sizes))), alpha=0.7, color='lightseagreen', edgecolor='black')
            plt.xlabel('Community Size')
            plt.ylabel('Frequency')
            plt.title('Community Size Distribution')
            plt.grid(True, alpha=0.3)

        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.85)
        fig2.suptitle('Causal Graph Analysis — Concepts and Communities', fontsize=16)
        plt.show()

    def _export_analysis_metrics(self, metrics, filename):
        """Export analysis metrics to JSON"""
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj

        # Remove complex objects that can't be serialized
        exportable_metrics = {}
        for category, data in metrics.items():
            if category == 'components':
                # Convert component sets to lists of node IDs
                exportable_metrics[category] = {
                    'largest_wcc_size': data['largest_wcc_size'],
                    'largest_scc_size': data['largest_scc_size'],
                    'wcc_sizes': data['wcc_sizes'],
                    'scc_sizes': data['scc_sizes'],
                    'num_weakly_connected_components': len(data['wcc_sizes']),
                    'num_strongly_connected_components': len(data['scc_sizes'])
                }
            else:
                exportable_metrics[category] = convert_numpy_types(data)

        # Add timestamp
        exportable_metrics['analysis_timestamp'] = datetime.now().isoformat()
        exportable_metrics['graph_id'] = f"merged_graph_{len(self.nodes)}_nodes"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(exportable_metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"Graph analysis metrics exported to {filename}")