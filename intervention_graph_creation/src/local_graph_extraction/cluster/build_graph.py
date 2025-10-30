import os
import pandas as pd
import networkx as nx

from config import load_settings

SETTINGS = load_settings()
output_dir = SETTINGS.paths.output_dir


def build_graph_from_csv(csv_dir: str) -> nx.Graph:
    """
    Build a NetworkX graph from CSV files.

    CSV folder should contain:
    - nodes_<label>.csv
    - edges_<type>.csv
    """
    G = nx.Graph()

    # Load nodes
    for file in os.listdir(csv_dir):
        if file.startswith("nodes_") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_dir, file))
            for _, row in df.iterrows():
                node_id = row["id"]
                label = row.get("label", "")
                attrs = row.to_dict()
                G.add_node(str(node_id), label=label, **attrs)

    # Load edges
    for file in os.listdir(csv_dir):
        if file.startswith("edges_") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_dir, file))
            edge_type = file.replace("edges_", "").replace(".csv", "")
            for _, row in df.iterrows():
                from_id = row["from_id"]
                to_id = row["to_id"]
                attrs = row.drop(["id", "from_id", "to_id"]).to_dict()
                attrs.pop("type", None)
                G.add_edge(str(from_id), str(to_id), type=edge_type, **attrs)

    return G


if __name__ == "__main__":
    csv_folder = os.path.join(output_dir, "processed_csv")
    G = build_graph_from_csv(csv_folder)
    print(
        f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )
