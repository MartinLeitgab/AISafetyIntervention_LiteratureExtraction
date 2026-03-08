import pickle
import networkx as nx
from pathlib import Path
import time
import networkit as nk

STEP1_DIR = Path("./phase2_results/step1_load_and_parse_umapwithoutlocalsatellites")

print("Loading existing data...")
with open(STEP1_DIR / "graph_node_attributes.pkl", "rb") as f:
    node_attrs = pickle.load(f)
with open(STEP1_DIR / "graph_edge_data.pkl", "rb") as f:
    edge_data = pickle.load(f)

print(f"Building graph from {len(node_attrs):,} nodes, {len(edge_data):,} edges...")
G = nx.DiGraph()
for node_id in node_attrs.keys():
    G.add_node(node_id)
for edge in edge_data:
    if edge["source"] in node_attrs and edge["target"] in node_attrs:
        G.add_edge(edge["source"], edge["target"])

print("\nGraph verification:")
print(f"  Nodes: {len(G.nodes()):,}")
print(f"  Edges: {len(G.edges()):,}")
print(f"  Is directed: {G.is_directed()}")
reciprocal = sum(1 for u, v in G.edges() if G.has_edge(v, u))
print(
    f"  Reciprocal edges: {reciprocal:,}/{len(G.edges()):,} ({100 * reciprocal / len(G.edges()):.1f}%)"
)

print("\nComputing betweenness with NetworKit (faster)...")
print("Estimated time: 2-5 hours")
start = time.time()

nk_graph = nk.nxadapter.nx2nk(G, weightAttr=None)
bc_calc = nk.centrality.Betweenness(nk_graph)
bc_calc.run()

node_list = list(G.nodes())
betweenness = {node_list[i]: bc_calc.score(i) for i in range(len(node_list))}

elapsed = time.time() - start
print(f"✓ Completed in {elapsed / 60:.1f} minutes")

for node_id, bc in betweenness.items():
    if node_id in node_attrs:
        node_attrs[node_id]["betweenness"] = bc

print("\nComputing degrees...")
for node_id in node_attrs:
    if node_id in G:
        node_attrs[node_id]["degree"] = G.degree(node_id)
        node_attrs[node_id]["in_degree"] = G.in_degree(node_id)
        node_attrs[node_id]["out_degree"] = G.out_degree(node_id)

print("Saving patched node attributes...")
with open(STEP1_DIR / "graph_node_attributes.pkl", "wb") as f:
    pickle.dump(node_attrs, f)

print("✓ Complete")
