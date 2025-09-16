# pip install falkordb networkx lxml
from falkordb import FalkorDB
import networkx as nx

GRAPH = "validation"
INCLUDE_EMBEDDING = False   # flip to True if you really want them

db = FalkorDB(host="localhost", port=6379)
g = db.select_graph(GRAPH)

# Fetch nodes (batch if very large)
res_nodes = g.query("""
MATCH (n)
RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props
""")

G = nx.DiGraph()

for row in res_nodes.result_set:
    if len(row) != 3:
        print(f"Malformed node row (expected 3 values): {row}")
        continue
    nid, labels, props = row
    props = props or {}
    if not INCLUDE_EMBEDDING:
        props.pop("embedding", None)
    # choose a display label
    props.setdefault("label", props.get("name") or str(nid))
    props["labels"] = labels
    G.add_node(nid, **props)

# Fetch edges
res_edges = g.query("""
MATCH (a)-[r]->(b)
RETURN id(a) AS src, id(b) AS dst, type(r) AS type, properties(r) AS props
""")

for row in res_edges.result_set:
    if len(row) != 4:
        print(f"Malformed edge row (expected 4 values): {row}")
        continue
    src, dst, rtype, rprops = row
    rprops = rprops or {}
    if not INCLUDE_EMBEDDING:
        rprops.pop("embedding", None)
    rprops["type"] = rtype
    G.add_edge(src, dst, **rprops)

# Write for Gephi (pick one)
nx.write_gexf(G, "graph.gexf")        # Gephi native
# nx.write_graphml(G, "graph.graphml")
print("Wrote graph.gexf")