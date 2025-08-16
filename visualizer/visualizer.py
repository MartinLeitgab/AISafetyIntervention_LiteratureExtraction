import json
from pyvis.network import Network

def create_graph_from_json(json_data, output_file="knowledge_graph.html"):
    net = Network(
        height="600px", 
        width="100%", 
        bgcolor="#222222", 
        font_color="white",
        directed=True
    )
    
    added_nodes = set()
    
    colors = {
        "concept": "#97C2FC",     
        "intervention": "#FFAB91", 
        "default": "#C5C5C5"       
    }
    
    # Process each logical chain
    for chain in json_data["logical_chains"]:
        chain_id = chain["chain_id"]
        
        # Add nodes
        for node in chain["nodes"]:
            node_id = node["id"]
            
            if node_id not in added_nodes:
                # Determine node color based on type
                node_type = node.get("type", "default")
                color = colors.get(node_type, colors["default"])
                
                # Create hover info
                hover_text = f"ID: {node_id}\\nType: {node_type}\\nTitle: {node['title']}\\nDescription: {node['description']}"
                
                # Add node with styling
                net.add_node(
                    node_id,
                    label=node["title"][:30] + "..." if len(node["title"]) > 30 else node["title"],  # Truncate long labels
                    title=hover_text,
                    color=color,
                    size=25
                )
                added_nodes.add(node_id)
        
        # Add edges
        for edge in chain["edges"]:
            source = edge["source_id"]
            target = edge["target_id"]
            
            # Create edge label and hover text
            edge_label = edge["title"]
            confidence = edge.get("confidence", "N/A")
            hover_text = f"Relationship: {edge_label}\\nConfidence: {confidence}\\nDescription: {edge['description']}"
            
            # Add edge with styling
            net.add_edge(
                source,
                target,
                label=edge_label,
                title=hover_text,
                color="#848484",
                width=2
            )
    
    net.set_options("""
    var options = {
        "physics": {
            "enabled": true,
            "stabilization": {"iterations": 200},
            "barnesHut": {
                "gravitationalConstant": -8000,     
                "centralGravity": 0.1,              
                "springLength": 200,                
                "springConstant": 0.02,             
                "damping": 0.09,
                "avoidOverlap": 1                   
            }
        },
        "nodes": {
            "font": {"size": 12},
            "margin": 10,                           
            "widthConstraint": {"maximum": 150}     
        },
        "edges": {
            "arrows": {"to": {"enabled": true}},
            "font": {"size": 10, "align": "middle"},
            "length": 250,                          
            "smooth": {
                "enabled": true,
                "type": "continuous",
                "roundness": 0.2
            }
        },
        "layout": {
            "improvedLayout": true,
            "clusterThreshold": 150                 
        }
    }
    """)
    

    html_string = net.generate_html()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_string)
    
    print(f"Graph saved as {output_file}")


if __name__ == "__main__":
    file = "2307.16513v2.pdf_o3.json"
    
    try:
        with open(file, "r") as f:
            json_data = json.load(f)
            create_graph_from_json(json_data)
    except Exception as e:
        print(f"Unexpected error: {e}")