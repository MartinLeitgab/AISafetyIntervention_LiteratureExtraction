import os
import json
import networkx as nx
from community import community_louvain  # Louvain method
from pyvis.network import Network
from collections import Counter
import plotly.graph_objects as go

from config import load_settings

SETTINGS = load_settings()
input_dir = SETTINGS.paths.input_dir
output_dir = SETTINGS.paths.output_dir


def collect_json_files(output_dir):
    json_files = []
    for root, dirs, files in os.walk(output_dir):
        if "embeddings" in dirs:
            dirs.remove("embeddings")
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    return json_files


def build_graph(json_paths):
    G = nx.Graph()

    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            for node in nodes:
                name = node["name"]
                G.add_node(name, **node)

            for edge in edges:
                src = edge["source_node"]
                tgt = edge["target_node"]
                if src in G.nodes and tgt in G.nodes:
                    weight = float(edge.get("edge_confidence", 1))
                    G.add_edge(src, tgt, weight=weight, **edge)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return G


def run_louvain_clustering(G):
    """Compute the best partition using Louvain Method"""
    partition = community_louvain.best_partition(G, weight="weight")
    return partition


def visualize_interactive(G, partition, filename="louvain_network.html"):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.force_atlas_2based()

    # Assign colors based on community
    community_colors = {}
    for node, comm in partition.items():
        if comm not in community_colors:
            community_colors[comm] = f"hsl({(comm * 70) % 360}, 80%, 60%)"
        net.add_node(node, title=node, color=community_colors[comm])

    for src, tgt, data in G.edges(data=True):
        net.add_edge(src, tgt, title=data.get("type", ""))

    net.show(filename, notebook=False)
    print(f"✅ Interactive graph saved to {filename}")


def visualize_top_communities(G, partition, top_k=5, filename="louvain_top5.html"):
    # Count community sizes
    community_sizes = Counter(partition.values())
    top_comms = [comm for comm, _ in community_sizes.most_common(top_k)]

    # Filter nodes belonging to top communities
    selected_nodes = [n for n, c in partition.items() if c in top_comms]
    subG = G.subgraph(selected_nodes).copy()

    # Initialize PyVis network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        notebook=False,
        directed=False,
    )

    # Add nodes with color grouping
    for node in subG.nodes():
        comm = partition[node]
        net.add_node(
            node,
            label=f"{node} (C{comm})",  # Label shows node + community ID
            group=comm,
        )

    # Add edges
    for u, v, data in subG.edges(data=True):
        net.add_edge(u, v, value=data.get("edge_confidence", 1))

    # Use a layout that spreads communities apart
    net.force_atlas_2based(
        gravity=-50, central_gravity=0.005, spring_length=150, spring_strength=0.08
    )

    # Save visualization
    net.show(filename, notebook=False)
    print(f"✅ Interactive PyVis visualization saved to: {filename}")


def visualize_top_communities_plotly(G, partition, top_k=5):
    """
    Visualize top_k largest Louvain communities using Plotly (interactive 2D layout).

    Parameters:
        G : networkx.Graph
            The full graph.
        partition : dict
            Precomputed Louvain partition {node: community_id}.
        top_k : int
            Number of largest communities to visualize.
    """
    # Identify top_k largest communities
    community_sizes = Counter(partition.values())
    top_comms = [comm for comm, _ in community_sizes.most_common(top_k)]

    # Subgraph with only top communities
    selected_nodes = [n for n, c in partition.items() if c in top_comms]
    subG = G.subgraph(selected_nodes).copy()

    # 2D layout
    pos = nx.spring_layout(subG, k=0.3, iterations=50, seed=42)

    # Prepare edge traces
    edge_x, edge_y = [], []
    for u, v in subG.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Prepare node traces
    node_x, node_y, node_color, node_text = [], [], [], []
    color_map = {
        comm: f"hsl({(i * 70) % 360},80%,50%)" for i, comm in enumerate(top_comms)
    }

    for node in subG.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        comm = partition[node]
        node_color.append(color_map.get(comm, "gray"))
        node_text.append(f"{node} (C{comm})")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(color=node_color, size=10, line_width=1),
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Top {top_k} Louvain Communities",
            title_x=0.5,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    fig.show()


if __name__ == "__main__":
    # Collect JSON files
    json_paths = collect_json_files(output_dir)
    print(f"Collected {len(json_paths)} JSON files from {output_dir}")

    # Build graph
    G = build_graph(json_paths)
    print(
        f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    # Run Louvain clustering
    partition = run_louvain_clustering(G)
    print(f"Detected {len(set(partition.values()))} communities")

    # Visualize interactively

    visualize_interactive(G, partition, filename="louvain_network.html")
    # visualize_top_communities(G, partition, top_k=5, filename="louvain_top5.html")
    # visualize_top_communities_plotly(G, partition, top_k=5)
