import json
import networkx as nx
import numpy as np
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class KnowledgeGraphVisualizer:
    def __init__(self, graph_file='mock_graph.json'):
        self.graph_file = graph_file
        self.data = None
        self.G = None
        self.node_positions = None
        self.load_graph_data()
        
    def load_graph_data(self):
        try:
            import os
            if not os.path.exists(self.graph_file):
                print(f"Error: {self.graph_file} file not found!")
                return
                
            file_size = os.path.getsize(self.graph_file)
            print(f"File size: {file_size} bytes")
            
            if file_size == 0:
                print(f"Error: {self.graph_file} file is empty!")
                return
                
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Loaded graph with {len(self.data['nodes'])} nodes")
            self.build_networkx_graph()
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Error loading graph data: {e}")
            
    def build_networkx_graph(self):
        self.G = nx.DiGraph()
        
        for node in self.data['nodes']:
            self.G.add_node(
                node['name'],
                **{k: v for k, v in node.items() if k != 'name'}
            )
        
        edge_count = 0
        for chain in self.data.get('logical_chains', []):
            for edge in chain['edges']:
                source = edge['source_node']
                target = edge['target_node']
                if source in self.G.nodes and target in self.G.nodes:
                    self.G.add_edge(
                        source, target,
                        type=edge['type'],
                        description=edge['description'],
                        confidence=edge.get('edge_confidence', 1)
                    )
                    edge_count += 1
        
        print(f"Built graph with {self.G.number_of_nodes()} nodes and {edge_count} edges")
        
    def compute_global_layout(self, layout_type='spring'):
        if layout_type == 'spring':
            self.node_positions = nx.spring_layout(self.G, k=3, iterations=50, seed=42)
        elif layout_type == 'kamada_kawai':
            self.node_positions = nx.kamada_kawai_layout(self.G)
        elif layout_type == 'circular':
            self.node_positions = nx.circular_layout(self.G)
        elif layout_type == 'shell':
            self.node_positions = nx.shell_layout(self.G)
        else:
            self.node_positions = nx.spring_layout(self.G, k=3, iterations=50, seed=42)
        
    def create_interactive_plotly_visualization(self, min_confidence: float = 0.0, label_top_k: int = 80):
        if self.node_positions is None:
            self.compute_global_layout()
            
        node_x = []
        node_y = []
        node_sizes = []
        node_colors = []
        node_text = []
        customdata = []
        
        categories = list(set([self.G.nodes[node].get('concept_category', 'Unknown') or 'Unknown' for node in self.G.nodes()]))
        category_to_idx = {cat: i for i, cat in enumerate(sorted(categories))}
        
        for node in self.G.nodes():
            pos = self.node_positions[node]
            node_x.append(pos[0])
            node_y.append(pos[1])
            
            d = self.G.degree(node)
            size = max(12, min(40, 15 + d * 3))
            node_sizes.append(size)
            
            category = self.G.nodes[node].get('concept_category', 'Unknown') or 'Unknown'
            node_colors.append(category_to_idx[category])
            
            description = self.G.nodes[node].get('description', 'No description') or 'No description'
            customdata.append([node, category, description])
        
        edge_x = []
        edge_y = []
        edges_detailed_info = []
        
        node_edge_info = {}
        for node in self.G.nodes():
            incoming_edges = []
            outgoing_edges = []
            
            for pred in self.G.predecessors(node):
                edge_data = self.G[pred][node]
                confidence = edge_data.get('confidence', edge_data.get('edge_confidence', 'N/A'))
                edge_type = edge_data.get('type', 'Unknown')
                description = edge_data.get('description', '')
                incoming_edges.append(f"← {pred}: {edge_type} (confidence: {confidence}) - {description}")
            
            for succ in self.G.successors(node):
                edge_data = self.G[node][succ]
                confidence = edge_data.get('confidence', edge_data.get('edge_confidence', 'N/A'))
                edge_type = edge_data.get('type', 'Unknown')
                description = edge_data.get('description', '')
                outgoing_edges.append(f"→ {succ}: {edge_type} (confidence: {confidence}) - {description}")
            
            edge_info = []
            if incoming_edges:
                edge_info.extend(["<b>Incoming:</b>"] + incoming_edges)
            if outgoing_edges:
                edge_info.extend(["<b>Outgoing:</b>"] + outgoing_edges)
            
            node_edge_info[node] = "<br>".join(edge_info) if edge_info else "No connections"
        
        for edge in self.G.edges():
            x0, y0 = self.node_positions[edge[0]]
            x1, y1 = self.node_positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edges_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1.5, color='rgba(100,100,100,0.6)'),
            hoverinfo='skip',
            showlegend=False,
            name='edges'
        )
        
        highlight_edges = go.Scatter(
            x=[],
            y=[],
            mode='lines',
            line=dict(width=3, color='crimson'),
            hoverinfo='skip',
            name='highlighted-edges',
            showlegend=False
        )
        
        edge_info_display = go.Scatter(
            x=[],
            y=[],
            mode='text',
            text=[],
            textposition="middle center",
            textfont=dict(size=10, color='red'),
            showlegend=False,
            name='edge-info'
        )
        
        top_nodes = set(sorted([(node, self.G.degree(node)) for node in self.G.nodes()], 
                              key=lambda x: x[1], reverse=True)[:max(0, label_top_k)])
        labels = [n if n in top_nodes else '' for n in self.G.nodes()]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=labels,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Concept Category Index"),
                opacity=0.9,
                line=dict(width=2, color='white')
            ),
            customdata=customdata,
            hovertemplate='<b>%{customdata[0]}</b><br>Category: %{customdata[1]}<br>Description: %{customdata[2]}<br><br>%{text}<extra></extra>',
            name='nodes'
        )
        
        enhanced_customdata = []
        for i, node in enumerate(self.G.nodes()):
            node_data = customdata[i]
            edge_info = node_edge_info[node]
            enhanced_customdata.append(node_data + [edge_info])
        
        node_trace.customdata = enhanced_customdata
        node_trace.hovertemplate = '<b>%{customdata[0]}</b><br>Category: %{customdata[1]}<br>Description: %{customdata[2]}<br><br><b>Connections:</b><br>%{customdata[3]}<extra></extra>'
        
        layout = go.Layout(
            title=dict(
                text=f"Interactive Knowledge Graph<br>{self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges<br><i>Click on nodes to see detailed edge information</i>",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=30, l=90, r=30, t=120),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig = go.Figure(data=[edges_trace, node_trace, highlight_edges, edge_info_display], layout=layout)
        
        nodes_list = list(self.G.nodes())
        node_positions_list = [[self.node_positions[node][0], self.node_positions[node][1]] for node in nodes_list]
        edges_data = []
        
        for node in nodes_list:
            node_edges = []
            for pred in self.G.predecessors(node):
                edge_data = self.G[pred][node]
                confidence = edge_data.get('confidence', edge_data.get('edge_confidence', 'N/A'))
                edge_type = edge_data.get('type', 'Unknown')
                description = edge_data.get('description', '')
                node_edges.append([pred, node, f"← {edge_type} (conf: {confidence}): {description}"])
            
            for succ in self.G.successors(node):
                edge_data = self.G[node][succ]
                confidence = edge_data.get('confidence', edge_data.get('edge_confidence', 'N/A'))
                edge_type = edge_data.get('type', 'Unknown')
                description = edge_data.get('description', '')
                node_edges.append([node, succ, f"→ {edge_type} (conf: {confidence}): {description}"])
            
            edges_data.append(node_edges)
        
        fig.update_layout(meta=dict(
            nodes=nodes_list,
            positions=node_positions_list,
            nodeEdges=edges_data
        ))
        
        output_file = 'knowledge_graph_interactive.html'
        script = """
        var gd = document.getElementById('kg2d');
        if (gd) {
            var meta = gd.layout.meta || {};
            var nodes = meta.nodes || [];
            var positions = meta.positions || [];
            var nodeEdges = meta.nodeEdges || [];
            
            function highlightNodeEdges(nodeIndex) {
                var ex = [], ey = [];
                var edges = nodeEdges[nodeIndex] || [];
                
                var nodePos = {};
                for (var i = 0; i < nodes.length; i++) {
                    nodePos[nodes[i]] = positions[i];
                }
                
                for (var i = 0; i < edges.length; i++) {
                    var edge = edges[i];
                    var source = edge[0];
                    var target = edge[1];
                    
                    if (nodePos[source] && nodePos[target]) {
                        ex.push(nodePos[source][0], nodePos[target][0], null);
                        ey.push(nodePos[source][1], nodePos[target][1], null);
                    }
                }
                
                Plotly.restyle(gd, {x: [ex], y: [ey]}, [2]);
                
                var infoText = edges.map(function(e) { return e[2]; }).join('\\n');
                console.log('Edge info for', nodes[nodeIndex], ':', infoText);
            }
            
            gd.on('plotly_click', function(eventData) {
                if (!eventData.points || !eventData.points.length) return;
                var point = eventData.points[0];
                if (point.curveNumber !== 1) return;
                
                var nodeIndex = point.pointIndex;
                highlightNodeEdges(nodeIndex);
            });
        }
        """
        
        html = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id='kg2d', post_script=script)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Enhanced interactive visualization saved as {output_file}")
        
        return fig
        
    def create_true_3d_visualization(self, width=1400, height=900):
        adj_matrix = nx.adjacency_matrix(self.G).todense()
        
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(adj_matrix)
        
        coords_3d = coords_3d * 5
        
        node_list = list(self.G.nodes())
        node_sizes = []
        node_colors_numeric = []
        node_hover_text = []
        
        categories = list(set([self.G.nodes[node].get('concept_category', 'Unknown') or 'Unknown' for node in self.G.nodes()]))
        category_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        for node in node_list:
            degree = self.G.degree(node)
            size = max(8, min(25, 10 + degree * 2))
            node_sizes.append(size)
            
            category = self.G.nodes[node].get('concept_category', 'Unknown') or 'Unknown'
            node_colors_numeric.append(category_to_idx[category])
            
            description = self.G.nodes[node].get('description', 'No description')
            hover_text = f"<b>{node}</b><br>Category: {category}<br>Degree: {degree}<br>Description: {description}"
            node_hover_text.append(hover_text)
        
        node_scatter3d = go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers+text',
            text=[node[:10] if len(node) > 10 else node for node in node_list],
            textposition="top center",
            textfont=dict(size=8),
            hoverinfo='text',
            hovertext=node_hover_text,
            marker=dict(
                size=node_sizes,
                color=node_colors_numeric,
                colorscale='viridis',
                opacity=0.8,
                line=dict(width=1, color='white'),
                showscale=True,
                colorbar=dict(title="Categories")
            ),
            name='nodes'
        )

        edges_by_confidence = {}
        
        for u, v in self.G.edges():
            i0 = node_list.index(u)
            i1 = node_list.index(v)
            
            edge_data = self.G[u][v]
            confidence = edge_data.get('confidence', edge_data.get('edge_confidence', 1))
            if confidence is None:
                confidence = 1
            
            conf_level = 'high' if confidence > 2 else 'medium' if confidence > 1 else 'low'
            if conf_level not in edges_by_confidence:
                edges_by_confidence[conf_level] = {'x': [], 'y': [], 'z': [], 'info': []}
            
            edges_by_confidence[conf_level]['x'].extend([coords_3d[i0, 0], coords_3d[i1, 0], None])
            edges_by_confidence[conf_level]['y'].extend([coords_3d[i0, 1], coords_3d[i1, 1], None])
            edges_by_confidence[conf_level]['z'].extend([coords_3d[i0, 2], coords_3d[i1, 2], None])
            
            edge_type = edge_data.get('type', 'Unknown')
            description = edge_data.get('description', '')
            info_text = f"{u} → {v}<br>Type: {edge_type}<br>Confidence: {confidence}<br>{description}"
            edges_by_confidence[conf_level]['info'].extend([info_text, info_text, None])

        edge_traces = []
        edge_colors = {'high': 'rgba(255,100,100,0.8)', 'medium': 'rgba(100,255,100,0.6)', 'low': 'rgba(100,100,255,0.4)'}
        edge_widths = {'high': 4, 'medium': 2, 'low': 1}
        
        for conf_level, edge_data in edges_by_confidence.items():
            if edge_data['x']:
                edge_trace = go.Scatter3d(
                    x=edge_data['x'],
                    y=edge_data['y'],
                    z=edge_data['z'],
                    mode='lines',
                    line=dict(width=edge_widths[conf_level], color=edge_colors[conf_level]),
                    hoverinfo='text',
                    hovertext=edge_data['info'],
                    name=f'{conf_level.capitalize()} confidence edges',
                    showlegend=True
                )
                edge_traces.append(edge_trace)

        data = edge_traces + [node_scatter3d]
        
        fig = go.Figure(data=data)
        
        fig.update_layout(
            title=dict(
                text=f"True 3D Knowledge Graph Visualization<br>{self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges",
                font=dict(size=20, color='white')
            ),
            scene=dict(
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False, 
                    showline=False, 
                    showbackground=False,
                    title=""
                ),
                yaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False, 
                    showline=False, 
                    showbackground=False,
                    title=""
                ),
                zaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False, 
                    showline=False, 
                    showbackground=False,
                    title=""
                ),
                aspectmode='cube',
                bgcolor='rgba(5,10,20,1)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1),
                    projection=dict(type='perspective')
                ),
                annotations=[]
            ),
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=80),
            paper_bgcolor='rgba(5,10,20,1)',
            plot_bgcolor='rgba(5,10,20,1)',
            font=dict(color='white'),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                font=dict(color='white')
            )
        )
        
        fig.update_scenes(
            dragmode='orbit',
            hovermode='closest'
        )
        
        output_file = 'knowledge_graph_true_3d.html'
        
        script = """
        var gd = document.getElementById('kg3d');
        if (gd) {
            var isRotating = false;
            
            function startRotation() {
                if (isRotating) return;
                isRotating = true;
                
                var angle = 0;
                
                function rotate() {
                    if (!isRotating) return;
                    
                    angle += 0.01;
                    var newCamera = {
                        eye: {
                            x: 2 * Math.cos(angle),
                            y: 2 * Math.sin(angle),
                            z: 1.5
                        },
                        center: {x: 0, y: 0, z: 0},
                        up: {x: 0, y: 0, z: 1}
                    };
                    
                    Plotly.relayout(gd, {'scene.camera': newCamera});
                    requestAnimationFrame(rotate);
                }
                rotate();
            }
            
            function stopRotation() {
                isRotating = false;
            }
            
            document.addEventListener('keydown', function(event) {
                if (event.key === 'r' || event.key === 'R') {
                    if (isRotating) stopRotation();
                    else startRotation();
                }
                if (event.key === 'c' || event.key === 'C') {
                    Plotly.relayout(gd, {
                        'scene.camera': {
                            eye: {x: 1.5, y: 1.5, z: 1.5},
                            center: {x: 0, y: 0, z: 0},
                            up: {x: 0, y: 0, z: 1}
                        }
                    });
                }
            });
            
            gd.on('plotly_relayout', function() {
                stopRotation();
            });
            
            var instructions = document.createElement('div');
            instructions.innerHTML = '<p style="color: white; position: absolute; top: 10px; left: 10px; z-index: 1000; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; font-size: 12px;">Press <b>R</b> to toggle rotation<br>Press <b>C</b> to reset camera<br>Drag to orbit, scroll to zoom</p>';
            document.body.appendChild(instructions);
        }
        """
        
        html = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id='kg3d', post_script=script)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"True 3D visualization saved as {output_file}")
        
        return fig
        
    def create_simple_3d_visualization(self, width=1400, height=900):
        pos_3d = nx.spring_layout(self.G, dim=3, k=3, iterations=50, seed=42)
        
        node_list = list(self.G.nodes())
        x_coords = [pos_3d[node][0] for node in node_list]
        y_coords = [pos_3d[node][1] for node in node_list]
        z_coords = [pos_3d[node][2] for node in node_list]
        
        node_sizes = []
        node_colors = []
        node_hover_text = []
        
        categories = list(set([self.G.nodes[node].get('concept_category', 'Unknown') or 'Unknown' for node in self.G.nodes()]))
        category_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        for node in node_list:
            degree = self.G.degree(node)
            size = max(8, min(25, 10 + degree * 2))
            node_sizes.append(size)
            
            category = self.G.nodes[node].get('concept_category', 'Unknown') or 'Unknown'
            node_colors.append(category_to_idx[category])
            
            description = self.G.nodes[node].get('description', 'No description')
            hover_text = f"<b>{node}</b><br>Category: {category}<br>Degree: {degree}<br>Description: {description}"
            node_hover_text.append(hover_text)
        
        node_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            hoverinfo='text',
            hovertext=node_hover_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='viridis',
                opacity=0.8,
                line=dict(width=1, color='white'),
                showscale=True,
                colorbar=dict(title="Categories")
            ),
            name='nodes'
        )

        edge_x = []
        edge_y = []
        edge_z = []
        edge_hover = []
        
        for u, v in self.G.edges():
            edge_data = self.G[u][v]
            edge_type = edge_data.get('type', 'Unknown')
            confidence = edge_data.get('confidence', edge_data.get('edge_confidence', 'N/A'))
            description = edge_data.get('description', '')
            
            edge_x.extend([pos_3d[u][0], pos_3d[v][0], None])
            edge_y.extend([pos_3d[u][1], pos_3d[v][1], None])
            edge_z.extend([pos_3d[u][2], pos_3d[v][2], None])
            
            hover_text = f"{u} → {v}<br>Type: {edge_type}<br>Confidence: {confidence}<br>{description}"
            edge_hover.extend([hover_text, hover_text, None])

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(width=2, color='rgba(150,150,150,0.6)'),
            hoverinfo='text',
            hovertext=edge_hover,
            name='edges',
            showlegend=False
        )

        highlight_trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines',
            line=dict(width=4, color='crimson'),
            hoverinfo='text',
            hovertext=[],
            name='highlighted',
            showlegend=False
        )

        fig = go.Figure(data=[edge_trace, node_trace, highlight_trace])
        
        fig.update_layout(
            title=dict(
                text=f"3D Knowledge Graph<br>{self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges",
                font=dict(size=20, color='white')
            ),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False, showbackground=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False, showbackground=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False, showbackground=False),
                bgcolor='rgba(5,10,20,1)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=80),
            paper_bgcolor='rgba(5,10,20,1)',
            plot_bgcolor='rgba(5,10,20,1)',
            font=dict(color='white')
        )
        
        fig.update_scenes(dragmode='orbit', hovermode='closest')
        
        output_file = 'knowledge_graph_3d.html'
        
        node_positions = {node: pos_3d[node] for node in node_list}
        
        script = """
        var gd = document.getElementById('kg3d');
        if (gd) {
            var nodePositions = """ + str(node_positions) + """;
            var nodes = """ + str(node_list) + """;
            
            gd.on('plotly_click', function(eventData) {
                if (!eventData.points || !eventData.points.length) return;
                
                var point = eventData.points[0];
                if (point.curveNumber !== 1) return;
                
                var clickedNode = nodes[point.pointIndex];
                var highlightX = [], highlightY = [], highlightZ = [], highlightText = [];
                
                var edges = """ + str(list(self.G.edges())) + """;
                
                for (var i = 0; i < edges.length; i++) {
                    var edge = edges[i];
                    if (edge[0] === clickedNode || edge[1] === clickedNode) {
                        var pos1 = nodePositions[edge[0]];
                        var pos2 = nodePositions[edge[1]];
                        
                        if (pos1 && pos2) {
                            highlightX.push(pos1[0], pos2[0], null);
                            highlightY.push(pos1[1], pos2[1], null);
                            highlightZ.push(pos1[2], pos2[2], null);
                            
                            var edgeData = """ + str([(u, v, self.G[u][v]) for u, v in self.G.edges()]) + """;
                            var edgeInfo = edgeData.find(function(e) { return e[0] === edge[0] && e[1] === edge[1]; });
                            if (edgeInfo) {
                                var hoverText = edge[0] + ' → ' + edge[1] + '<br>Type: ' + (edgeInfo[2].type || 'Unknown') + '<br>Confidence: ' + (edgeInfo[2].confidence || edgeInfo[2].edge_confidence || 'N/A');
                                highlightText.push(hoverText, hoverText, null);
                            } else {
                                highlightText.push('', '', null);
                            }
                        }
                    }
                }
                
                Plotly.restyle(gd, {
                    x: [highlightX],
                    y: [highlightY], 
                    z: [highlightZ],
                    text: [highlightText]
                }, [2]);
            });
        }
        """
        
        html = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id='kg3d', post_script=script)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"3D visualization saved as {output_file}")
        
        return fig
        
    def create_statistics_dashboard(self):
        stats = {
            'Total Nodes': self.G.number_of_nodes(),
            'Total Edges': self.G.number_of_edges(),
            'Average Degree': np.mean([self.G.degree(node) for node in self.G.nodes()]),
            'Density': nx.density(self.G),
            'Connected Components': nx.number_connected_components(self.G.to_undirected()),
            'Isolated Nodes': len(list(nx.isolates(self.G))),
            'Self Loops': self.G.number_of_selfloops() if hasattr(self.G, 'number_of_selfloops') else 0
        }
        
        try:
            original_nodes = len(self.data.get('nodes', []))
        except Exception:
            original_nodes = self.G.number_of_nodes()
        stats['Original Node Entries'] = original_nodes
        stats['Unique Nodes (by name)'] = self.G.number_of_nodes()
        
        categories = Counter([self.G.nodes[node].get('concept_category', 'Unknown') or 'Unknown'
                            for node in self.G.nodes()])
        
        edge_types = Counter([self.G[u][v].get('type', 'Unknown') 
                            for u, v in self.G.edges()])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Graph Statistics', 'Concept Categories', 'Edge Types', 'Degree Distribution'),
            specs=[[{"type": "table"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[k for k in stats.keys()], 
                                 [f"{v:.2f}" if isinstance(v, float) else str(v) for v in stats.values()]])
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=list(categories.keys()),
                y=list(categories.values()),
                name='Concept Categories'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=list(edge_types.keys()),
                y=list(edge_types.values()),
                name='Edge Types'
            ),
            row=2, col=1
        )
        
        degrees = [self.G.degree(node) for node in self.G.nodes()]
        fig.add_trace(
            go.Histogram(
                x=degrees,
                nbinsx=20,
                name='Degree Distribution'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Knowledge Graph Statistics Dashboard",
            height=800,
            showlegend=False
        )
        
        output_file = 'knowledge_graph_stats.html'
        fig.write_html(output_file)
        print(f"Statistics dashboard saved as {output_file}")
        
        return fig
        
    def run_visualizations(self):
        print("Starting enhanced knowledge graph visualization...")
        print("=" * 60)
        
        print("Creating enhanced interactive visualization...")
        try:
            self.create_interactive_plotly_visualization()
        except Exception as e:
            print(f"Error in interactive visualization: {e}")
        
        print("Creating true 3D visualization...")
        try:
            self.create_true_3d_visualization()
        except Exception as e:
            print(f"Error in true 3D visualization: {e}")
        
        print("Creating simple 3D visualization...")
        try:
            self.create_simple_3d_visualization()
        except Exception as e:
            print(f"Error in simple 3D visualization: {e}")
        
        print("Creating statistics dashboard...")
        try:
            self.create_statistics_dashboard()
        except Exception as e:
            print(f"Error in statistics dashboard: {e}")
        
        print("\n" + "=" * 60)
        print("Enhanced visualization complete!")
        print("Generated files:")
        print("- knowledge_graph_interactive.html (with detailed edge information on hover and click)")
        print("- knowledge_graph_true_3d.html (true 3D with confidence-based edge coloring)")
        print("- knowledge_graph_3d.html (simple 3D visualization, crash-proof)")
        print("- knowledge_graph_stats.html (comprehensive statistics)")
        print("\nInteraction tips:")
        print("- 2D: Hover over nodes to see all edge details, click to highlight connections")
        print("- 3D: Press 'R' to toggle auto-rotation, 'C' to reset camera, drag to orbit")

def main():
    print("Enhanced Knowledge Graph Visualizer")
    print("=" * 60)
    
    visualizer = KnowledgeGraphVisualizer()
    
    if visualizer.G is None:
        print("Failed to load graph data. Exiting.")
        return
    
    visualizer.run_visualizations()

if __name__ == "__main__":
    main()