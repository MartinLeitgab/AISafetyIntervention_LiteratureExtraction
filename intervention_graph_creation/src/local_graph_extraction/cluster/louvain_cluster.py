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
    """Compute the best partition using Louvain Method"""

    partition = community_louvain.best_partition(G, weight="weight")
    return partition


def visualize_interactive(G, partition):
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
