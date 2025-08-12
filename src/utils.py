"""Shared utilities for working with concept graphs and visualization."""

import os
import matplotlib.pyplot as plt
import networkx as nx

from src.models.concept_graph import ConceptGraph


def _graph_to_networkx(graph: ConceptGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    # Add nodes
    for node in graph.nodes:
        G.add_node(
            node.id,
            label=node.name,
            is_intervention=bool(node.is_intervention),
        )
    # Add edges
    for e in graph.edges:
        label = getattr(e.relationship, "value", e.relationship)
        conf = getattr(e.confidence, "value", e.confidence)
        for s in e.source_node_ids:
            for t in e.target_node_ids:
                if s in G and t in G and s != t:
                    # Last label wins if duplicate; acceptable for visualization
                    G.add_edge(s, t, label=label, confidence=conf)
    return G


def _save_graph_png(G: nx.DiGraph, out_path: str) -> None:
    if not G.nodes:
        # Create an empty placeholder figure
        plt.figure(figsize=(6, 4), dpi=200)
        plt.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return

    plt.figure(figsize=(10, 8), dpi=200)
    pos = nx.spring_layout(G, seed=42)  # deterministic layout

    # Colors by intervention flag
    node_colors = ["#f94144" if G.nodes[n].get("is_intervention") else "#90caf9" for n in G.nodes]
    # Node labels
    node_labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
    # Edge labels
    edge_labels = nx.get_edge_attributes(G, "label")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors="#333333", linewidths=0.8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.2, connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5, rotate=False)

    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


