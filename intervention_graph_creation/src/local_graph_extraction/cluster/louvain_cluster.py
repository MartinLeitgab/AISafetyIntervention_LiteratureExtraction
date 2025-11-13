import os
import random
import networkx as nx
from community import community_louvain
from collections import Counter
from pyvis.network import Network

from build_graph import build_graph_from_csv
from config import load_settings

SETTINGS = load_settings()
output_dir = SETTINGS.paths.output_dir


def run_louvain_clustering(G: nx.Graph):
    """
    Perform community detection on a given graph using the Louvain method.

    This function computes an optimal partition of nodes that maximizes modularity,
    using the specified edge weight (default: "weight") to determine connection strength.

    Args:
        G (nx.Graph): The input NetworkX graph (can be weighted or unweighted).

    Returns:
        dict: A mapping of each node to its assigned community ID.
    """

    partition = community_louvain.best_partition(G, weight="weight")
    return partition


def visualize_interactive(G, partition):
    """
    Create an interactive PyVis visualization of the Louvain communities.

    Each node is colored according to its detected community, and all edges
    in the graph are displayed. The result is saved as an HTML file that
    can be opened in a web browser.

    Args:
        G (nx.Graph): The input NetworkX graph.
        partition (dict): A dictionary mapping nodes to community IDs.

    Output:
        Generates an interactive HTML file named 'louvain_clusters.html'.
    """
    net = Network(height="800px", width="100%", notebook=False, directed=True)
    community_colors = {}

    for comm in set(partition.values()):
        community_colors[comm] = "#%06x" % random.randint(0, 0xFFFFFF)

    for node, comm in partition.items():
        node_str = str(node)
        net.add_node(node_str, title=node_str, color=community_colors[comm])

    for u, v in G.edges():
        net.add_edge(str(u), str(v))

    net.show("louvain_clusters.html", notebook=False)


def visualize_top_communities(
    G: nx.Graph, partition: dict, top_k=5, filename="louvain_top5.html"
):
    """
    Visualize the top-k largest Louvain communities using PyVis.

    This function selects the 'top_k' largest communities based on node count,
    extracts the corresponding subgraph, and produces an interactive visualization
    highlighting their internal structure.

    Args:
        G (nx.Graph): The input NetworkX graph.
        partition (dict): Mapping of nodes to community IDs.
        top_k (int, optional): Number of largest communities to visualize. Defaults to 5.
        filename (str, optional): Name of the output HTML file. Defaults to 'louvain_top5.html'.

    Output:
        Saves an interactive visualization (HTML) of the top-k communities to disk.
    """
    community_sizes = Counter(partition.values())
    top_comms = [comm for comm, _ in community_sizes.most_common(top_k)]
    selected_nodes = [n for n, c in partition.items() if c in top_comms]
    subG = G.subgraph(selected_nodes).copy()

    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    for node in subG.nodes():
        comm = partition[node]
        net.add_node(node, label=f"{node} (C{comm})", group=comm)
    for u, v, data in subG.edges(data=True):
        net.add_edge(u, v, value=data.get("edge_confidence", 1))
    net.force_atlas_2based(
        gravity=-50, central_gravity=0.005, spring_length=150, spring_strength=0.08
    )
    net.show(filename, notebook=False)
    print(f"✅ Top {top_k} communities visualization saved to: {filename}")


if __name__ == "__main__":
    csv_folder = os.path.join(output_dir, "processed_csv")
    G = build_graph_from_csv(csv_folder)

    partition = run_louvain_clustering(G)
    # visualize_interactive(G, partition)
    visualize_top_communities(G, partition)
