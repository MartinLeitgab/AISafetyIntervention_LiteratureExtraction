import networkx as nx
from input_handler.input_handler import create_sample_graph_data, validate_graph_data
from graph_merger.graph_merger import EnhancedGlobalCausalGraph
from metrics_analyzer.metrics_analyzer import MetricsAnalyzer
from visualizer.visualizer import Visualizer
from utilities.utils import MATPLOTLIB_AVAILABLE, logger
from models.models import CausalNode, CausalEdge
def test_enhanced_merger_with_visuals():
    """Test the enhanced merger with full visualization"""
    print("Testing Enhanced Causal Graph Merger with Visualizations...")
    
    # Create and validate test data
    sample_graphs = create_sample_graph_data()
    if not validate_graph_data(sample_graphs):
        print("❌ Invalid graph data")
        return
    
    # Visualize individual local graphs
    print("\n📊 Visualizing Local Graphs:")
    visualizer = None  # Will be initialized after merging
    for i, local_graph in enumerate(sample_graphs, 1):
        title = f"Local Graph {i}: {local_graph.get('title', 'Unknown')}"
        nodes_count = len(local_graph.get('nodes', []))
        edges_count = len(local_graph.get('edges', []))
        print(f"  - {title} (Nodes: {nodes_count}, Edges: {edges_count})")
        for j, node in enumerate(local_graph.get('nodes', []), 1):
            print(f"    Node {j}: {node['concept_text']} (Intervention: {node.get('isIntervention', 0)})")
        if MATPLOTLIB_AVAILABLE:
            visualizer = Visualizer(nx.DiGraph(), {}, {})
            visualizer.visualize_local_graph(local_graph, title)
    
    # Initialize and merge graphs
    merger = EnhancedGlobalCausalGraph(similarity_threshold=0.75)
    try:
        print("\n🔄 Merging graphs...")
        merger.merge_local_graphs(sample_graphs)
        print("✅ Merge completed successfully!")
        
        # Get quality metrics
        quality_metrics = merger.get_merge_quality_metrics()
        print(f"\nMerge Quality Metrics:")
        for category, metrics in quality_metrics.items():
            print(f"  {category}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.3f}")
        
        # Visualize merged result
        print("\n📊 Visualizing Merged Global Graph:")
        visualizer = Visualizer(merger.graph, merger.nodes, merger.edges)
        if merger.nodes and MATPLOTLIB_AVAILABLE:
            print(f"Final merged graph contains {len(merger.nodes)} nodes:")
            for node_id, node in merger.nodes.items():
                sources = len(node.source_papers)
                print(f"  - {node.concept_text} (ID: {node_id}, Sources: {sources}, Intervention: {node.isIntervention})")
            visualizer.visualize_with_edge_details(show_edge_labels=True, figsize=(16, 12))
        else:
            print("No nodes in merged graph to visualize or matplotlib not available")
        
        # Run comprehensive analysis
        print("\n" + "="*60)
        print("🔍 RUNNING COMPREHENSIVE GRAPH ANALYSIS")
        print("="*60)
        analyzer = MetricsAnalyzer(merger.graph, merger.nodes, merger.edges)
        analysis_results = analyzer.analyze_graph_metrics(
            export_filename="graph_analysis_metrics.json",
            show_plots=MATPLOTLIB_AVAILABLE
        )
        
        # Export results
        export_data = merger.export_enhanced_json("test_enhanced_merge.json")
        print(f"\n💾 Exported merged graph to test_enhanced_merge.json")
        
        return merger
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        raise

def test_core_merger_logic():
    """Test the core merger logic without visualization"""
    print("Testing Core Enhanced Causal Graph Merger Logic...")
    
    sample_graphs = create_sample_graph_data()
    if not validate_graph_data(sample_graphs):
        print("❌ Invalid graph data")
        return
    
    merger = EnhancedGlobalCausalGraph(similarity_threshold=0.75)
    try:
        merger.merge_local_graphs(sample_graphs)
        print("✅ Merge completed successfully!")
        
        quality_metrics = merger.get_merge_quality_metrics()
        print(f"\nMerge Quality Metrics:")
        for category, metrics in quality_metrics.items():
            print(f"  {category}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.3f}")
        
        return merger
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        raise

if __name__ == "__main__":
    if MATPLOTLIB_AVAILABLE:
        print("🎨 Running enhanced merger with full visualizations...")
        test_merger = test_enhanced_merger_with_visuals()
    else:
        print("⚙️ Running core merger logic (no visualizations)...")
        test_merger = test_core_merger_logic()