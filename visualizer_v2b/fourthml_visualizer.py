
import json
from pyvis.network import Network

def create_graph_from_json(json_data, output_file="knowledge_graph.html"):
    net = Network(
        height="800px", 
        width="100%", 
        bgcolor="#FFFFFF", 
        font_color="black",
        directed=True
    )
    
    added_nodes = set()
    
    # Colors for different node types
    colors = {
        "concept": "#97C2FC",     
        "intervention": "#FFAB91", 
        "default": "#C5C5C5"       
    }
    
    # Build graph structure for topological sorting and connected components
    all_nodes = set()
    edges = []
    in_degree = {}
    out_degree = {}
    adjacency = {}  # For finding connected components
    
    # Collect all nodes and edges
    for node in json_data["nodes"]:
        node_name = node["name"]
        all_nodes.add(node_name)
        in_degree[node_name] = 0
        out_degree[node_name] = 0
        adjacency[node_name] = set()
    
    for chain in json_data["logical_chains"]:
        for edge in chain["edges"]:
            source = edge["source_node"]
            target = edge["target_node"]
            edges.append((source, target, chain["title"], edge))
            in_degree[target] = in_degree.get(target, 0) + 1
            out_degree[source] = out_degree.get(source, 0) + 1
            # Build undirected adjacency for connected components
            adjacency[source].add(target)
            adjacency[target].add(source)
    
    # Find connected components (separate chains)
    def find_connected_components():
        visited = set()
        components = []
        
        for node in all_nodes:
            if node not in visited:
                component = set()
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        stack.extend(adjacency[current] - visited)
                
                components.append(component)
        
        return components
    
    connected_components = find_connected_components()
    
    # Topological sort within each component with better layer balancing
    def topological_sort_component(component_nodes):
        layers = []
        remaining_nodes = component_nodes.copy()
        temp_in_degree = {node: sum(1 for s, t, _, _ in edges if s in component_nodes and t == node) for node in component_nodes}
        
        while remaining_nodes:
            # Find nodes with no incoming edges within this component
            current_layer = []
            for node in remaining_nodes:
                if temp_in_degree.get(node, 0) == 0:
                    current_layer.append(node)
            
            if not current_layer:
                # Handle cycles by taking nodes with minimum in-degree
                min_in_degree = min(temp_in_degree.get(node, 0) for node in remaining_nodes)
                current_layer = [node for node in remaining_nodes if temp_in_degree.get(node, 0) == min_in_degree]
            
            layers.append(current_layer)
            
            # Remove current layer nodes and update in-degrees
            for node in current_layer:
                remaining_nodes.remove(node)
                # Reduce in-degree for all targets of this node within component
                for source, target, _, _ in edges:
                    if source == node and target in remaining_nodes and target in component_nodes:
                        temp_in_degree[target] -= 1
        
        return layers
    
    # Advanced edge crossing minimization using barycenter method
    def minimize_crossings(component_layers, component_edges):
        """Reorder nodes within layers to minimize edge crossings"""
        if len(component_layers) < 2:
            return component_layers
            
        optimized_layers = [layer.copy() for layer in component_layers]
        component_edge_dict = {}
        
        # Build adjacency for this component
        for source, target, _, _ in component_edges:
            if source not in component_edge_dict:
                component_edge_dict[source] = []
            component_edge_dict[source].append(target)
        
        # Iterate through layers and optimize ordering
        for iteration in range(3):  # Multiple passes for better optimization
            for layer_idx in range(1, len(optimized_layers)):
                current_layer = optimized_layers[layer_idx]
                prev_layer = optimized_layers[layer_idx - 1]
                
                # Calculate barycenter for each node in current layer
                node_barycenters = []
                for node in current_layer:
                    # Find predecessors in previous layer
                    predecessors = []
                    for i, prev_node in enumerate(prev_layer):
                        if prev_node in component_edge_dict and node in component_edge_dict[prev_node]:
                            predecessors.append(i)
                    
                    # Calculate barycenter (average position of predecessors)
                    if predecessors:
                        barycenter = sum(predecessors) / len(predecessors)
                    else:
                        barycenter = len(prev_layer)  # Put nodes with no predecessors at the end
                    
                    node_barycenters.append((node, barycenter))
                
                # Sort by barycenter
                node_barycenters.sort(key=lambda x: x[1])
                optimized_layers[layer_idx] = [node for node, _ in node_barycenters]
        
        return optimized_layers
    
    # Calculate layout for all components with crossing minimization
    component_layouts = []
    max_layers = 0
    
    for component in connected_components:
        if component:  # Skip empty components
            # Get edges for this component
            component_edges = [(s, t, c, e) for s, t, c, e in edges if s in component and t in component]
            
            # Get initial topological layers
            layers = topological_sort_component(component)
            
            # Optimize layer ordering to minimize crossings
            optimized_layers = minimize_crossings(layers, component_edges)
            
            component_layouts.append(optimized_layers)
            max_layers = max(max_layers, len(optimized_layers))
    
    # Adaptive layout parameters based on graph size
    def calculate_layout_params(component_layouts):
        max_nodes_per_layer = 1
        total_layers = 0
        
        for component_layers in component_layouts:
            for layer in component_layers:
                max_nodes_per_layer = max(max_nodes_per_layer, len(layer))
            total_layers = max(total_layers, len(component_layers))
        
        # Adaptive spacing based on complexity
        base_layer_height = 180
        base_node_width = 280
        base_component_spacing = 700
        
        # Increase spacing for denser graphs
        layer_height = base_layer_height + min(50, max_nodes_per_layer * 10)
        node_width = base_node_width + min(100, max_nodes_per_layer * 20)
        component_spacing = base_component_spacing + min(200, len(component_layouts) * 50)
        
        return layer_height, node_width, component_spacing
    
    layer_height, node_width, component_spacing = calculate_layout_params(component_layouts)
    
    # Create a mapping from node names to node data
    node_data = {node["name"]: node for node in json_data["nodes"]}
    
    # Add all nodes with hierarchical and component-separated positioning
    for component_index, component_layers in enumerate(component_layouts):
        # Calculate component's base X position
        component_base_x = component_index * component_spacing
        
        # Calculate the width of this component to center it
        max_nodes_in_layer = max(len(layer) for layer in component_layers) if component_layers else 0
        component_width = (max_nodes_in_layer - 1) * node_width if max_nodes_in_layer > 1 else 0
        component_center_offset = -component_width / 2
        
        for layer_index, layer_nodes in enumerate(component_layers):
            y_pos = layer_index * layer_height
            
            # Sort nodes in layer: interventions last, concepts first
            layer_nodes_sorted = sorted(layer_nodes, key=lambda n: (
                node_data.get(n, {}).get("type", "default") == "intervention",  # interventions last
                node_data.get(n, {}).get("type", "default") != "concept"        # concepts first
            ))
            
            for node_index, node_name in enumerate(layer_nodes_sorted):
                if node_name not in added_nodes:
                    node = node_data.get(node_name, {"name": node_name, "type": "default", "description": "Unknown"})
                    node_type = node.get("type", "default")
                    color = colors.get(node_type, colors["default"])
                    
                    # Calculate position within component
                    layer_width = (len(layer_nodes_sorted) - 1) * node_width
                    local_x = node_index * node_width - layer_width / 2
                    x_pos = component_base_x + local_x
                    
                    # Build comprehensive display text with all node attributes
                    display_parts = [f"{node_name}"]
                    
                    # Add aliases if they exist
                    if node.get("aliases"):
                        aliases_str = ", ".join(node["aliases"])
                        display_parts.append(f"Aliases: {aliases_str}")
                    
                    # Add concept-specific attributes
                    if node.get("concept_category"):
                        display_parts.append(f"Category: {node['concept_category']}")
                    
                    # Add intervention-specific attributes
                    if node.get("intervention_lifecycle"):
                        display_parts.append(f"Lifecycle: {node['intervention_lifecycle']}")
                    
                    if node.get("intervention_maturity"):
                        display_parts.append(f"Maturity: {node['intervention_maturity']}")
                    
                    # Add description
                    display_parts.append(f"Desc: {node['description']}")
                    
                    # Create label with line breaks (use \n for actual line breaks)
                    label = "\n".join(display_parts)
                    
                    # Create hover text with proper HTML line breaks
                    hover_parts = [
                        f"Name: {node_name}",
                        f"Type: {node_type}",
                        f"Description: {node['description']}",
                        f"Component: {component_index + 1}",
                        f"Layer: {layer_index}"
                    ]
                    
                    if node.get("aliases"):
                        aliases_str = ", ".join(node["aliases"])
                        hover_parts.append(f"Aliases: {aliases_str}")
                    
                    if node.get("concept_category"):
                        hover_parts.append(f"Category: {node['concept_category']}")
                    
                    if node.get("intervention_lifecycle"):
                        hover_parts.append(f"Lifecycle: {node['intervention_lifecycle']}")
                    
                    if node.get("intervention_maturity"):
                        hover_parts.append(f"Maturity: {node['intervention_maturity']}")
                    
                    hover_text = "<br>".join(hover_parts)
                    
                    net.add_node(
                        node_name,
                        label=label,
                        title=hover_text,
                        color=color,
                        size=30,
                        x=x_pos,
                        y=y_pos,
                        physics=False  # Fixed positioning
                    )
                    added_nodes.add(node_name)
    
    # Add edges from logical chains with advanced overlap minimization
    edge_positions = {}  # Track edge routing to avoid overlaps
    
    for source, target, chain_title, edge_data in edges:
        relationship_type = edge_data["type"]
        
        # Find node positions
        source_pos = None
        target_pos = None
        source_layer = None
        target_layer = None
        source_component = None
        target_component = None
        source_layer_index = None
        target_layer_index = None
        
        # Find positions and layers
        for comp_idx, component_layers in enumerate(component_layouts):
            for layer_idx, layer_nodes in enumerate(component_layers):
                if source in layer_nodes:
                    source_layer = layer_idx
                    source_component = comp_idx
                    source_layer_index = layer_nodes.index(source)
                if target in layer_nodes:
                    target_layer = layer_idx
                    target_component = comp_idx
                    target_layer_index = layer_nodes.index(target)
        
        # Calculate optimal edge routing
        edge_key = f"{source_component}_{source_layer}_{target_layer}"
        if edge_key not in edge_positions:
            edge_positions[edge_key] = 0
        edge_positions[edge_key] += 1
        parallel_offset = edge_positions[edge_key]
        
        # Build edge display text with metadata
        edge_display_parts = [
            relationship_type,
            f"Conf: {edge_data['edge_confidence']}"
        ]
        
        edge_label = "\n".join(edge_display_parts)
        
        # Build edge hover text with all edge attributes (use HTML breaks)
        edge_hover_parts = [
            f"Chain: {chain_title}",
            f"Relationship: {relationship_type}",
            f"Description: {edge_data['description']}",
            f"Confidence: {edge_data['edge_confidence']}"
        ]
        
        edge_hover_text = "<br>".join(edge_hover_parts)
        
        # Smart edge styling based on positioning and conflicts
        if source_layer is not None and target_layer is not None:
            layer_distance = abs(target_layer - source_layer)
            
            # Calculate edge curvature to avoid overlaps
            if layer_distance == 1:
                # Adjacent layers - minimal curve, offset for parallel edges
                roundness = 0.1 + (parallel_offset - 1) * 0.1
                edge_color = "#666666"
                edge_width = max(1.5, 3 - parallel_offset * 0.2)
            elif layer_distance == 2:
                # Skip one layer - moderate curve
                roundness = 0.3 + (parallel_offset - 1) * 0.1
                edge_color = "#777777"
                edge_width = max(1.5, 2.5 - parallel_offset * 0.2)
            else:
                # Long distance - high curve to go over intermediate nodes
                roundness = 0.6 + (parallel_offset - 1) * 0.1
                edge_color = "#888888"
                edge_width = max(1.5, 2 - parallel_offset * 0.1)
            
            # Add horizontal offset for same-layer parallel edges
            if source_layer_index is not None and target_layer_index is not None:
                horizontal_spread = abs(source_layer_index - target_layer_index)
                if horizontal_spread == 0:
                    # Same position - increase curve significantly
                    roundness = min(0.8, roundness + 0.3)
        else:
            # Fallback
            roundness = 0.3
            edge_color = "#999999"
            edge_width = 2
        
        net.add_edge(
            source,
            target,
            label=edge_label,
            title=edge_hover_text,
            color=edge_color,
            width=edge_width,
            smooth={"type": "continuous", "roundness": roundness},
            font={"align": "top", "size": 8, "vadjust": -10}
        )
    
    # Configure network physics and layout options
    net.set_options("""
    var options = {
        "physics": {
            "enabled": false
        },
        "nodes": {
            "font": {"size": 10, "multi": "html", "align": "center"},
            "margin": 15,                           
            "widthConstraint": {"maximum": 220},
            "heightConstraint": {"maximum": 160}     
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 1.2}},
            "font": {
                "size": 8, 
                "align": "top",
                "vadjust": -10,
                "strokeWidth": 2,
                "strokeColor": "#ffffff"
            },
            "length": 350,
            "smooth": {
                "enabled": true,
                "type": "continuous"
            },
            "labelHighlightBold": true
        },
        "layout": {
            "randomSeed": 42
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "selectConnectedEdges": false
        }
    }
    """)
    
    # Generate and save HTML
    html_string = net.generate_html()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_string)
    
    print(f"Graph saved as {output_file}")




if __name__ == "__main__":
    file = "2307.166513v2_sample.json"  # Updated default filename
    
    try:
        with open(file, "r") as f:
            json_data = json.load(f)
            create_graph_from_json(json_data)
    except FileNotFoundError:
        print(f"File {file} not found. Please ensure the JSON file exists.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except KeyError as e:
        print(f"Missing required key in JSON structure: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")