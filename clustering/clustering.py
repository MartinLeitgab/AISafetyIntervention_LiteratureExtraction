import networkx as nx
import itertools

def get_clusters_girvan_newman(graph, num_clusters, most_valuable_edge=None):
    """ Get a cluster structure of a graph using the Girvan-Newman algorithm.
    Args:
        graph (networkx.Graph): The input graph.
        num_clusters (int): The number of clusters to find. (num_clusters > 1)
        most_valuable_edge (optional, callable): The method to find the most valuable edge.
            girvan_newman uses edge_betweenness_centrality by default (None)
    Returns:
        A list of lists: each list contains nodes for each cluster.
    """
    # if graph is empty 
    if graph.number_of_nodes() == 0 or graph.number_of_nodes() == 1:
        raise ValueError("Your graph is empty or has only one node")

    if num_clusters <= 1:
        raise ValueError("num_clusters must be greater than 1")

    # the number of nodes in the graph
    if num_clusters > graph.number_of_nodes():
        raise ValueError("num_clusters must be less than or equal to the number of nodes in the graph")

    comp = nx.community.girvan_newman(graph, most_valuable_edge=most_valuable_edge)

    # get num_clusters clusters
    limited = itertools.takewhile(lambda c: len(c) <= num_clusters, comp)
    for cluster_structure in limited:
        if len(cluster_structure)== num_clusters:
            return [list(c) for c in cluster_structure]

