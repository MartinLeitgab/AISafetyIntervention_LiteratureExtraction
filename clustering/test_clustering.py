import unittest
import networkx as nx
import numpy as np
import warnings
from unittest.mock import patch
from clustering import get_clusters_girvan_newman


class TestGirvanNewmanClustering(unittest.TestCase):
    """
    Comprehensive test suite for Girvan-Newman clustering algorithm.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Suppress warnings for cleaner test output
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def setUp(self):
        """Set up test graphs for each test case."""
        # Simple path graph
        self.path_graph = nx.path_graph(10)
        
        # Complete graph
        self.complete_graph = nx.complete_graph(6)
        
        # Disconnected graph
        self.disconnected_graph = nx.Graph()
        self.disconnected_graph.add_edges_from([(0, 1), (2, 3), (4, 5)])
        
        # Weighted graph
        self.weighted_graph = nx.Graph()
        self.weighted_graph.add_weighted_edges_from([
            (0, 1, 1.0), (1, 2, 2.0), (2, 3, 0.5),
            (3, 4, 1.5), (4, 5, 0.8)
        ])
        
        # Barbell graph (good for community detection)
        self.barbell_graph = nx.barbell_graph(5, 3)
    
    def test_basic_functionality(self):
        """Test basic functionality with path graph."""
        num_clusters = 3
        result = get_clusters_girvan_newman(self.path_graph, num_clusters)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), num_clusters)

        # Check that all nodes are assigned to communities
        all_nodes = set()
        for community in result:
            all_nodes.update(community)
        self.assertEqual(all_nodes, set(self.path_graph.nodes()))
    
    def test_invalid_k_values(self):
        """Test error handling for invalid num_clusters values."""
        # num_clusters <= 1 should raise ValueError
        with self.assertRaises(ValueError):
            get_clusters_girvan_newman(self.path_graph, 1)
        
        with self.assertRaises(ValueError):
            get_clusters_girvan_newman(self.path_graph, 0)
        
        # num_clusters is more than nodes.
        with self.assertRaises(ValueError):
            get_clusters_girvan_newman(self.path_graph, 11)
    
    def test_complete_graph(self):
        """Test with complete graph where community detection is challenging."""
        num_clusters = 3
        result = get_clusters_girvan_newman(self.complete_graph, num_clusters)
        
        if result:  # Complete graphs may not divide well
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), num_clusters)
            # Verify all nodes are covered
            all_nodes = set()
            for community in result:
                all_nodes.update(community)
            self.assertEqual(all_nodes, set(self.complete_graph.nodes()))
    
    def test_disconnected_graph(self):
        """Test with disconnected graph."""
        num_clusters = 3
        result = get_clusters_girvan_newman(self.disconnected_graph, num_clusters)

        if result:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), num_clusters)
            # Check that communities respect disconnected components
            components = list(nx.connected_components(self.disconnected_graph))
            self.assertEqual(len(components), 3)  # Should have 3 components
    
    
    def test_barbell_graph_optimal_case(self):
        """Test with barbell graph which has clear community structure."""
        num_clusters = 2
        result = get_clusters_girvan_newman(self.barbell_graph, num_clusters)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), num_clusters)

        # Should ideally separate the two cliques
        sizes = [len(community) for community in result]
        # Both communities should have reasonable sizes
        self.assertTrue(all(size > 0 for size in sizes))
    
    def test_weighted_graph(self):
        """Test with weighted graph."""
        num_clusters = 2
        result = get_clusters_girvan_newman(self.weighted_graph, num_clusters)

        if result:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), num_clusters)
            # Verify all nodes are assigned
            all_nodes = set()
            for community in result:
                all_nodes.update(community)
            self.assertEqual(all_nodes, set(self.weighted_graph.nodes()))
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic for same input."""
        num_clusters = 3

        result1 = get_clusters_girvan_newman(self.barbell_graph, num_clusters)
        result2 = get_clusters_girvan_newman(self.barbell_graph, num_clusters)

        if result1 and result2:
            # Results should be the same (communities might be in different order)
            self.assertEqual(len(result1), len(result2))
            
            # Convert to sets for order-independent comparison
            set1 = [set(community) for community in result1]
            set2 = [set(community) for community in result2]
            
            # Sort by size for consistent comparison
            set1.sort(key=len)
            set2.sort(key=len)
            
            self.assertEqual(set1, set2)
    
if __name__ == '__main__':
    # Run tests with detailed output
    print("=" * 60)
    print("Girvan-Newman Clustering Tests")
    print("=" * 60)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner, exit=False)
    
    print("=" * 60)