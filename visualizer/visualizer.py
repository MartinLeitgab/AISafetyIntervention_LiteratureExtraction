# visualizer.py
import networkx as nx
from typing import Dict, List
import matplotlib.pyplot as plt
from models.models import CausalNode, CausalEdge
from utilities.utils import MATPLOTLIB_AVAILABLE
from collections import defaultdict

class Visualizer:
    """Handles visualization of graphs and metrics"""
    
    def __init__(self, graph: nx.DiGraph, nodes: Dict[str, CausalNode], edges: Dict[str, List[CausalEdge]]):
        self.graph = graph
        self.nodes = nodes
        self.edges = edges

    def visualize_with_edge_details(self, figsize=(20, 16), show_edge_labels=False):
        """Enhanced visualization with optional edge labels"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create visualization.")
            return
            
        plt.figure(figsize=figsize)
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=3, iterations=150, seed=42)
        
        # Enhanced node styling
        node_colors = []
        node_sizes = []
        node_alphas = []
        
        for node_id in self.graph.nodes():
            node = self.nodes[node_id]
            
            # Size by evidence strength (sources + connections)
            evidence_strength = len(node.source_papers) + self.graph.degree(node_id)
            node_sizes.append(800 + evidence_strength * 200)
            
            # Alpha by confidence (ensure it stays within 0-1 range)
            alpha_value = 0.6 + 0.4 * min(1.0, node.confidence_score)
            node_alphas.append(min(1.0, max(0.3, alpha_value)))
            
            # Color by type and implementation
            if node.isIntervention == 1:
                if node.implemented == 1:
                    node_colors.append('darkgreen')
                elif node.maturity_level and node.maturity_level >= 3:
                    node_colors.append('mediumseagreen')
                else:
                    node_colors.append('lightgreen')
            else:
                # Categorize problems by keywords
                keywords = set(node.semantic_keywords)
                if {'deception', 'hacking', 'misalignment'} & keywords:
                    node_colors.append('darkred')
                elif {'oversight', 'evaluation', 'scalability'} & keywords:
                    node_colors.append('orange')
                elif {'interpretability', 'detection', 'monitoring'} & keywords:
                    node_colors.append('gold')
                else:
                    node_colors.append('lightblue')
        
        # Draw nodes with varying transparency
        for i, node_id in enumerate(self.graph.nodes()):
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[node_id],
                                 node_color=[node_colors[i]], node_size=[node_sizes[i]],
                                 alpha=node_alphas[i], edgecolors='black', linewidths=1.5)
        
        # Enhanced edge drawing with relationship types
        self._draw_relationship_edges(pos)
        
        # Node labels with evidence indicators
        labels = {}
        for node_id in self.graph.nodes():
            node = self.nodes[node_id]
            concept = node.concept_text
            evidence_count = len(node.source_papers)
            
            # Truncate long labels and add evidence indicator
            if len(concept) > 30:
                words = concept.split()
                label = ' '.join(words[:4]) + '...' if len(words) > 4 else concept
            else:
                label = concept
            
            if evidence_count > 1:
                label += f" [{evidence_count}]"
            
            labels[node_id] = label
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, font_weight='bold')
        
        # Optional edge labels (for smaller graphs)
        if show_edge_labels and len(self.graph.edges()) < 50:
            edge_labels = {}
            for edge_key, edge_list in list(self.edges.items())[:20]:  # Limit to prevent clutter
                if edge_list:
                    primary_edge = edge_list[0]
                    source_target = edge_key.split(':')[0]
                    if '→' in source_target:
                        source, target = source_target.split('→')
                        if self.graph.has_edge(source, target):
                            rel_type = primary_edge.relationship_type
                            confidence = primary_edge.confidence
                            edge_labels[(source, target)] = f"{rel_type}\n(conf:{confidence})"
            
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)
        
        plt.title("Enhanced AI Safety Causal Knowledge Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Enhanced legend
        self._draw_comprehensive_legend()
        
        plt.tight_layout()
        plt.show()
    

    def _draw_relationship_edges(self, pos):
        """Draw edges with relationship-specific styling"""
        relationship_styles = {
            'causes': {'color': 'darkblue', 'style': 'solid', 'alpha': 0.8},
            'prevents': {'color': 'red', 'style': 'dashed', 'alpha': 0.7},
            'enables': {'color': 'green', 'style': 'solid', 'alpha': 0.7},
            'moderates': {'color': 'purple', 'style': 'dotted', 'alpha': 0.6},
            'necessitates': {'color': 'orange', 'style': 'solid', 'alpha': 0.8},
            'correlates': {'color': 'gray', 'style': 'dotted', 'alpha': 0.5}
        }
        
        # Group edges by relationship type and confidence
        relationship_edges = defaultdict(lambda: defaultdict(list))
        
        for edge_key, edge_list in self.edges.items():
            if edge_list:
                primary_edge = edge_list[0]
                source_target = edge_key.split(':')[0]
                if '→' in source_target:
                    source, target = source_target.split('→')
                    if self.graph.has_edge(source, target):
                        rel_type = primary_edge.relationship_type
                        confidence = primary_edge.confidence
                        relationship_edges[rel_type][confidence].append((source, target))
        
        # Draw each relationship type
        for rel_type, confidence_groups in relationship_edges.items():
            style = relationship_styles.get(rel_type, relationship_styles['causes'])
            
            for confidence, edge_list in confidence_groups.items():
                width = max(1, confidence * 0.8)
                alpha = style['alpha'] * (0.5 + 0.1 * confidence)
                
                nx.draw_networkx_edges(
                    self.graph, pos, edgelist=edge_list,
                    edge_color=style['color'], width=width,
                    alpha=alpha, arrows=True, arrowsize=20,
                    arrowstyle='->', style=style['style']
                )
    

    def _draw_comprehensive_legend(self):
        """Draw comprehensive legend with all visual elements"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = [
            # Node types
            Patch(facecolor='darkgreen', label='Implemented Solutions'),
            Patch(facecolor='mediumseagreen', label='Mature Solutions'),
            Patch(facecolor='lightgreen', label='Proposed Solutions'),
            Patch(facecolor='darkred', label='Critical AI Safety Problems'),
            Patch(facecolor='orange', label='Oversight & Evaluation Challenges'),
            Patch(facecolor='gold', label='Interpretability & Detection'),
            Patch(facecolor='lightblue', label='Other Concepts'),
            
            # Divider
            Line2D([0], [0], color='white', linewidth=0, label=''),
            
            # Relationship types
            Line2D([0], [0], color='darkblue', linewidth=3, label='Causal Relationships'),
            Line2D([0], [0], color='red', linewidth=3, linestyle='--', label='Prevention/Mitigation'),
            Line2D([0], [0], color='green', linewidth=3, label='Enabling Relationships'),
            Line2D([0], [0], color='purple', linewidth=2, linestyle=':', label='Moderation'),
            Line2D([0], [0], color='orange', linewidth=3, label='Necessity'),
            Line2D([0], [0], color='gray', linewidth=1, linestyle=':', label='Correlation'),
        ]
        
        # Add note about evidence indicators
        legend_elements.append(Line2D([0], [0], color='white', linewidth=0, 
                                    label='[n] = Evidence from n papers'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
                  frameon=True, fancybox=True, shadow=True, fontsize=9)
    

    def visualize_local_graph(self, local_graph_data: dict, title: str = "Local Graph"):
        """Visualize a single local graph"""
        if not MATPLOTLIB_AVAILABLE:
            print(f"Cannot visualize {title} - matplotlib not available")
            return

        temp_graph = nx.DiGraph()
        node_colors = []
        node_labels = {}

        for node_data in local_graph_data.get('nodes', []):
            node_id = node_data['concept_text']
            temp_graph.add_node(node_id)
            node_colors.append('lightgreen' if node_data.get('isIntervention', 0) == 1 else 'lightcoral')
            node_labels[node_id] = node_id[:22] + "..." if len(node_id) > 25 else node_id

        for edge_data in local_graph_data.get('edges', []):
            for source in edge_data['source_nodes']:
                for target in edge_data['target_nodes']:
                    temp_graph.add_edge(source, target)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(temp_graph, seed=42)
        nx.draw_networkx_nodes(temp_graph, pos, node_color=node_colors, node_size=2000, alpha=0.8, edgecolors='black')
        nx.draw_networkx_edges(temp_graph, pos, edge_color='darkblue', arrows=True, arrowsize=20, alpha=0.6)
        nx.draw_networkx_labels(temp_graph, pos, node_labels, font_size=10, font_weight='bold')
        plt.title(f"{title}\nNodes: {len(temp_graph.nodes())}, Edges: {len(temp_graph.edges())}", fontsize=14, fontweight='bold')
        plt.axis('off')

        if MATPLOTLIB_AVAILABLE:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightgreen', label='Intervention Nodes'),
                Patch(facecolor='lightcoral', label='Problem/Concept Nodes')
            ]
            plt.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()
