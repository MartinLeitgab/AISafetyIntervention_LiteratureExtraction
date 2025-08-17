import json
import graphviz
import html
import os

def render_json_graph(json_file_path, json_dir = None):
    """
    Render a JSON knowledge graph using Graphviz.
    
    Args:
        json_file_path (str): Path to the JSON file containing graph data
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{json_file_path}' not found. Please check the file path.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
        return

    graph_name = data.get("paper_title", "Unnamed Graph").replace(" ", "_").replace(":", "")
    dot = graphviz.Digraph(comment=data.get("paper_title", "Unnamed Graph"), strict=False)

    # Configure graph attributes for better readability
    dot.attr(rankdir='TB')          # Top-to-Bottom layout
    dot.attr(ranksep='1.5')         # Vertical separation between ranks
    dot.attr(nodesep='0.8')         # Horizontal separation between nodes
    dot.attr(splines='true')        # Default splines for proper edge label support
    dot.attr(concentrate='true')    # Merge parallel edges
    
    # Configure font settings for better readability
    dot.attr(fontname='Arial', fontsize='24')
    dot.attr('node', 
             shape='box', 
             style='filled,rounded', 
             fontname='Arial', 
             fontsize='18',
             margin='0.3,0.2',
             width='3',
             height='1.5')
    dot.attr('edge', 
             fontname='Arial', 
             fontsize='16',
             color='gray40',
             penwidth='2')

    # First pass: collect all nodes and create them with proper attributes
    all_nodes = {}
    chain_first_nodes = []
    
    for chain_idx, chain in enumerate(data.get("logical_chains", [])):
        chain_id = chain.get("chain_id", f"Unknown_Chain_{chain_idx}")
        chain_description = chain.get("description", "")
        nodes = chain.get("nodes", [])
        
        if not nodes:
            continue
            
        for i, node_data in enumerate(nodes):
            node_id = node_data.get("id", f"node_{i}")
            unique_node_id = f"{chain_id}_{node_id}" if not node_id.startswith(chain_id) else node_id
            
            node_type = node_data.get("type", "concept")
            node_title = node_data.get("title", "No Title")
            raw_node_description = node_data.get("description", "")
            node_maturity = node_data.get("maturity")
            
            escaped_node_description = html.escape(raw_node_description)
            fillcolor = 'lightblue' if node_type == 'concept' else 'lightgreen'
            
            # Create node label with chain header for first node
            if i == 0:
                node_label_html = f'''<
                    <B><FONT COLOR="darkblue" POINT-SIZE="22">{html.escape(chain_id)}</FONT></B><BR/>
                    <I><FONT POINT-SIZE="16">{html.escape(chain_description)}</FONT></I><BR/><BR/>
                    <B><FONT POINT-SIZE="20">{html.escape(node_title)}</FONT></B><BR/>
                    <FONT POINT-SIZE="18">{escaped_node_description}</FONT>
                '''
                chain_first_nodes.append(unique_node_id)
            else:
                node_label_html = f'''<
                    <B><FONT POINT-SIZE="20">{html.escape(node_title)}</FONT></B><BR/>
                    <FONT POINT-SIZE="18">{escaped_node_description}</FONT>
                '''
            
            # Add maturity data for interventions
            if node_type == 'intervention' and node_maturity is not None:
                node_label_html += f'<BR/><FONT COLOR="darkred" POINT-SIZE="16">Maturity: {node_maturity}</FONT>'
            node_label_html += '>'

            # Store node information
            all_nodes[unique_node_id] = {
                'chain_idx': chain_idx,
                'node_idx': i,
                'label': node_label_html,
                'fillcolor': fillcolor,
                'original_id': node_id
            }

    # Create all nodes in the graph
    for unique_node_id, node_info in all_nodes.items():
        dot.node(unique_node_id, label=node_info['label'], fillcolor=node_info['fillcolor'])

    # Second pass: create only explicit edges
    for chain_idx, chain in enumerate(data.get("logical_chains", [])):
        chain_id = chain.get("chain_id", f"Unknown_Chain_{chain_idx}")
        edges = chain.get("edges", [])
        
        for edge_data in edges:
            source_id = edge_data.get("source_id")
            target_id = edge_data.get("target_id")
            edge_title = edge_data.get("title", "")
            edge_confidence = edge_data.get("confidence")

            # Ensure we're using the correct unique IDs
            source_unique = f"{chain_id}_{source_id}" if not source_id.startswith(chain_id) else source_id
            target_unique = f"{chain_id}_{target_id}" if not target_id.startswith(chain_id) else target_id
            
            # Only create edge if both nodes exist
            if source_unique in all_nodes and target_unique in all_nodes:
                edge_label = edge_title if edge_title else "Connection"
                if edge_confidence is not None:
                    if edge_title:
                        edge_label += f' (Conf: {edge_confidence})'
                    else:
                        edge_label = f'Confidence: {edge_confidence}'
                
                dot.edge(source_unique, target_unique, xlabel=edge_label, weight='5')
                print(f"Created explicit edge: {source_unique} -> {target_unique} with label: {edge_label}")
            else:
                print(f"Warning: Edge references non-existent nodes: {source_id} -> {target_id}")
        
        print(f"Chain {chain_id} has {len(edges)} explicit edges")

    # Create invisible edges between chains for vertical layout ordering
    if len(chain_first_nodes) > 1:
        for i in range(len(chain_first_nodes) - 1):
            dot.edge(chain_first_nodes[i], chain_first_nodes[i + 1], 
                    style='invis', weight='100')

    # Render the graph
    # json_filename = f"{graph_name}_graph"
    json_filename = os.path.join(json_dir, f"{graph_name}_graph")
    dot.render(json_filename, view=True, format='svg', cleanup=True)
    print(f"Graph rendered and saved as '{json_filename}.svg'")
    
    # Also create a PNG version
    dot.render(json_filename + '_png', view=False, format='png', cleanup=True)
    print(f"PNG version saved as '{json_filename}_png.png'")


if __name__ == "__main__":
    json_file = "graph_data.json"
    render_json_graph(json_file)
