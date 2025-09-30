# Needs update to new schema - not working currently (09/30/25)
# Is this utility needed at all or is it deprecated?

# import os
# import json
# import networkx as nx

# def sanitize(value):
#     """Convert None or non-strings to safe exportable values for GEXF."""
#     if value is None:
#         return ""
#     if isinstance(value, (list, dict)):
#         return json.dumps(value, ensure_ascii=False)
#     return str(value)

# def _merge_aliases(existing_aliases_str, new_aliases_list):
#     existing = [a.strip() for a in existing_aliases_str.split(",")] if existing_aliases_str else []
#     new = [a.strip() for a in (new_aliases_list or [])]
#     merged = sorted({a for a in existing + new if a})
#     return ", ".join(merged)

# def merge_local_graphs_to_gexf(input_dir, output_file):
#     #  MultiDiGraph so we don't lose parallel edges with different relations/descriptions
#     G = nx.MultiDiGraph()
    
#     total_files = 0
#     # Goes through each json in each subfolder in processed directory 
#     for root, _, files in os.walk(input_dir):
#         for fname in files:
#             if not fname.lower().endswith(".json"):
#                 continue
#             total_files += 1
#             json_path = os.path.join(root, fname)
#             try:
#                 with open(json_path, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#             except Exception as e:
#                 print(f"Skipping {json_path} (read error: {e})")
#                 continue

#             # Nodes 
#             for node in data.get("nodes", []):
#                 node_id = sanitize(node.get("name"))
#                 if not node_id:
#                     continue

#                 attrs = {
#                     "aliases": sanitize(", ".join(node.get("aliases", []))),
#                     "type": sanitize(node.get("type")),  
#                     "description": sanitize(node.get("description")),
#                     "concept_category": sanitize(node.get("concept_category")),
#                     "intervention_lifecycle": sanitize(node.get("intervention_lifecycle")),
#                     "intervention_maturity": sanitize(node.get("intervention_maturity")),
#                 }

#                 if G.has_node(node_id):
#                     # merge aliases; fill empty attrs only
#                     existing = G.nodes[node_id]
#                     existing["aliases"] = _merge_aliases(existing.get("aliases", ""), node.get("aliases", []))
#                     for k, v in attrs.items():
#                         if not existing.get(k) and v:
#                             existing[k] = v
#                 else:
#                     G.add_node(node_id, **attrs)

#             # Edges from logical_chains 
#             for chain in data.get("logical_chains", []):
#                 for edge in chain.get("edges", []):
#                     src = sanitize(edge.get("source_node"))
#                     tgt = sanitize(edge.get("target_node"))
#                     if not src or not tgt:
#                         continue

#                     # ensure endpoints exist even if absent in nodes list
#                     if not G.has_node(src):
#                         G.add_node(src)
#                     if not G.has_node(tgt):
#                         G.add_node(tgt)

#                     # rename edge 'type' -> 'relation'
#                     G.add_edge(
#                         src,
#                         tgt,
#                         relation=sanitize(edge.get("type")),
#                         description=sanitize(edge.get("description")),
#                         confidence=sanitize(edge.get("edge_confidence")),
#                     )

#     nx.write_gexf(G, output_file)
#     print(f" Processed {total_files} JSON files")
#     print(f" Exported merged graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to {output_file}")

# if __name__ == "__main__":
#     merge_local_graphs_to_gexf(
#         "processed",
#         "merged_local_graphs.gexf"
#     )
