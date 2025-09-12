import json
import networkx as nx


def sanitize(value):
    """Convert None or non-strings to safe exportable values for GEXF."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def convert_single_json_to_gexf(json_file, output_file):
    # Load JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.MultiDiGraph()

    #  Graph-level metadata (from "meta")
    for meta in data.get("meta", []):
        key = sanitize(meta.get("key"))
        value = sanitize(meta.get("value"))
        if key:
            G.graph[key] = value

    #  Nodes
    for node in data.get("nodes", []):
        node_id = sanitize(node.get("name"))
        if not node_id:
            continue
        G.add_node(
            node_id,
            aliases=sanitize(", ".join(node.get("aliases", []))),
            type=sanitize(node.get("type")),
            description=sanitize(node.get("description")),
            concept_category=sanitize(node.get("concept_category")),
            intervention_lifecycle=sanitize(node.get("intervention_lifecycle")),
            intervention_maturity=sanitize(node.get("intervention_maturity")),
        )

    # Edges from logical_chains
    for chain in data.get("logical_chains", []):
        for edge in chain.get("edges", []):
            src = sanitize(edge.get("source_node"))
            tgt = sanitize(edge.get("target_node"))
            if not src or not tgt:
                continue

            if not G.has_node(src):
                G.add_node(src)
            if not G.has_node(tgt):
                G.add_node(tgt)

            G.add_edge(
                src,
                tgt,
                relation=sanitize(edge.get("type")),  # renamed from "type"
                description=sanitize(edge.get("description")),
                confidence=sanitize(edge.get("edge_confidence")),
                chain_title=sanitize(chain.get("title")),
            )

    # --- Export to GEXF ---
    nx.write_gexf(G, output_file)
    print(f" Exported {json_file} → {output_file}")
    print(f" Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


if __name__ == "__main__":
    convert_single_json_to_gexf(
        # using already processed json file in 'processed' directory
        r"processed\arxiv__arxiv_org_abs_1105_3821\arxiv__arxiv_org_abs_1105_3821.json",
        r"single_graph.gexf",
    )
