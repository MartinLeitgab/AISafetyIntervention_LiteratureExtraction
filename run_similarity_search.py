#!/usr/bin/env python
"""
Script to demonstrate using the similarity search functionality
with the FalkorDB graph database.

Usage:
    python run_similarity_search.py [--port PORT] [--query QUERY] [--threshold THRESHOLD]
"""

import argparse
import json

from intervention_graph_creation.src.local_graph_extraction.db.similarity_search import \
    SimilaritySearch


def parse_args():
    parser = argparse.ArgumentParser(description="Run similarity search on FalkorDB")
    parser.add_argument("--port", type=int, default=6379, 
                        help="FalkorDB port (default: 6379)")
    parser.add_argument("--query", type=str, 
                        default="AI alignment methods for large language models",
                        help="Text query to search for")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Similarity threshold (0-1)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to return")
    parser.add_argument("--max-cluster-size", type=int, default=10,
                        help="Maximum number of nodes in a cluster (default: 10)")
    parser.add_argument("--mode", choices=["search", "clusters", "neighborhoods", "merges"], 
                        default="search", help="Operation mode (search, clusters, neighborhoods, merges)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file to write JSON results (optional)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Connecting to FalkorDB on port {args.port}")
    search = SimilaritySearch(port=args.port)
    
    results = None  # Variable to store results for JSON output
    
    if args.mode == "search":
        try:
            print(f"\nSearching for nodes similar to: '{args.query}'")
            similar_nodes = search.find_similar_nodes(
                text=args.query, 
                top_k=args.top_k, 
                threshold=args.threshold
            )
            
            results = similar_nodes  # Store for potential JSON output
            
            if similar_nodes:
                print(f"Found {len(similar_nodes)} similar nodes:")
                for i, node in enumerate(similar_nodes):
                    print(f"{i+1}. {node['name']} (Similarity: {node['similarity']:.3f})")
                    print(f"   Type: {node['type']}")
                    description = node['description']
                    if description:
                        print(f"   Description: {description[:100]}..." if len(description) > 100 else f"   Description: {description}")
                    
                    # Display literature/sources if available
                    literature = node.get('literature', [])
                    if literature:
                        print(f"   Literature/Sources ({len(literature)}):")
                        for j, lit in enumerate(literature[:3]):  # Show up to 3 sources
                            title = lit.get('title', 'Untitled')
                            authors = lit.get('authors', '')
                            year = lit.get('year', '')
                            url = lit.get('url', '')
                            
                            author_year = f"{authors} ({year})" if authors and year else authors or year
                            print(f"     - {title}{': ' + author_year if author_year else ''}")
                            if url:
                                print(f"       {url}")
                        
                        # If there are more sources, show a message
                        if len(literature) > 3:
                            print(f"       ... and {len(literature) - 3} more sources")
                    
                    print()  # Add blank line between nodes
            else:
                print("No similar nodes found. Try lowering the threshold or adding more nodes to the database.")
        except Exception as e:
            print(f"Error during search: {e}")
            print("This could be due to FalkorDB connection issues or missing vector search capabilities.")
            print("Try restarting Docker and ensure FalkorDB is properly configured.")
            
    elif args.mode == "clusters":
        try:
            print(f"\nFinding clusters with similarity threshold {args.threshold} (max size: {args.max_cluster_size})...")
            clusters = search.find_clusters(
                threshold=args.threshold, 
                min_cluster_size=2, 
                max_cluster_size=args.max_cluster_size
            )
            
            results = clusters  # Store for potential JSON output
            
            if clusters:
                print(f"Found {len(clusters)} clusters:")
                for i, cluster in enumerate(clusters):
                    print(f"\nCluster {i+1} with {len(cluster)} nodes:")
                    for j, node in enumerate(cluster):
                        print(f"  {j+1}. {node['name']} (Similarity: {node['similarity']:.3f})")
                        if j == 0:  # Print more details for the seed node
                            print(f"     Type: {node['type']}")
                            description = node['description']
                            if description:
                                print(f"     Description: {description[:100]}..." if len(description) > 100 else f"     Description: {description}")
            else:
                print("No clusters found. Try lowering the threshold or adding more nodes to the database.")
        except Exception as e:
            print(f"Error finding clusters: {e}")
            print("This could be due to FalkorDB connection issues or missing vector search capabilities.")
            print("Try restarting Docker and ensure FalkorDB is properly configured.")
    
    elif args.mode == "neighborhoods":
        try:
            print(f"\nFinding clusters with neighborhood information (threshold: {args.threshold}, max size: {args.max_cluster_size})...")
            enriched_clusters = search.get_cluster_with_neighborhoods(
                threshold=args.threshold, 
                max_cluster_size=args.max_cluster_size
            )
            
            results = enriched_clusters  # Store for potential JSON output
            
            if enriched_clusters:
                print(f"Found {len(enriched_clusters)} clusters with neighborhood information:")
                for i, cluster in enumerate(enriched_clusters):
                    print(f"\nCluster {i+1} with {len(cluster)} nodes:")
                    for j, enriched_node in enumerate(cluster):
                        print(f"  {j+1}. {enriched_node['name']} (Similarity: {enriched_node['similarity']:.3f})")
                        
                        # Print neighborhood summary
                        neighborhood = enriched_node.get('neighborhood', {})
                        edge_count = len(neighborhood.get('edges', []))
                        neighbor_count = len(neighborhood.get('neighbors', {}))
                        print(f"     Neighborhood: {edge_count} edges, {neighbor_count} neighbors")
                        
                        # Print a sample of edges if available
                        if edge_count > 0:
                            print("     Sample edges:")
                            for k, edge in enumerate(neighborhood.get('edges', [])[:3]):  # Show up to 3 edges
                                print(f"       - {edge.get('type', 'UNKNOWN')} → {edge.get('target', 'UNKNOWN')}")
                            
                            if edge_count > 3:
                                print(f"       ... and {edge_count - 3} more edges")
            else:
                print("No clusters with neighborhood information found. Try lowering the threshold.")
        except Exception as e:
            print(f"Error finding clusters with neighborhoods: {e}")
            print("This could be due to FalkorDB connection issues or missing vector search capabilities.")
            print("Try restarting Docker and ensure FalkorDB is properly configured.")
    
    elif args.mode == "merges":
        try:
            print(f"\nSuggesting nodes to merge with threshold {args.threshold} (max cluster size: {args.max_cluster_size})...")
            merges = search.suggest_merges(
                threshold=args.threshold,
                max_cluster_size=args.max_cluster_size
            )
            
            results = merges  # Store for potential JSON output
            
            if merges:
                print(f"Found {len(merges)} potential merges:")
                for i, merge in enumerate(merges):
                    primary = merge['primary_node']
                    candidates = merge['merge_candidates']
                    print(f"\nMerge {i+1}: Keep '{primary['name']}' and merge:")
                    
                    # Display primary node details
                    print(f"  Primary: {primary['name']} ({primary['type']})")
                    if primary.get('description'):
                        desc = primary['description']
                        print(f"    Description: {desc[:100]}..." if len(desc) > 100 else f"    Description: {desc}")
                    
                    # Show neighborhood summary for primary node
                    neighborhood = primary.get('neighborhood', {})
                    edge_count = len(neighborhood.get('edges', []))
                    neighbor_count = len(neighborhood.get('neighbors', {}))
                    print(f"    Connections: {edge_count} edges, {neighbor_count} neighbors")
                    
                    # Display merge candidates
                    print("\n  Merge candidates:")
                    for j, candidate in enumerate(candidates):
                        print(f"    {j+1}. '{candidate['name']}' (Similarity: {candidate['similarity']:.3f})")
                        
                        # Show neighborhood summary for candidate
                        c_neighborhood = candidate.get('neighborhood', {})
                        c_edge_count = len(c_neighborhood.get('edges', []))
                        c_neighbor_count = len(c_neighborhood.get('neighbors', {}))
                        print(f"      Connections: {c_edge_count} edges, {c_neighbor_count} neighbors")
                    
                    # Ask if user wants to execute this merge
                    should_merge = input("\nDo you want to execute this merge? (y/n): ")
                    if should_merge.lower() == 'y':
                        for candidate in candidates:
                            try:
                                success = search.execute_merge(primary['name'], candidate['name'])
                                if success:
                                    print(f"  ✓ Merged '{candidate['name']}' into '{primary['name']}'")
                                else:
                                    print(f"  ✗ Failed to merge '{candidate['name']}' into '{primary['name']}'")
                            except Exception as merge_error:
                                print(f"  ✗ Error while merging: {merge_error}")
            else:
                print("No merge suggestions found. Try lowering the threshold.")
        except Exception as e:
            print(f"Error finding merge suggestions: {e}")
            print("This could be due to FalkorDB connection issues or missing vector search capabilities.")
            print("Try restarting Docker and ensure FalkorDB is properly configured.")
    
    # Save results to JSON file if requested
    if args.output and results:
        try:
            import json
            
            # Create a custom JSON encoder to handle non-serializable types
            class CustomJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    # Handle any special types here
                    try:
                        # Try to convert to a Python built-in type
                        return json.JSONEncoder.default(self, obj)
                    except TypeError:
                        # If that fails, convert to string
                        return str(obj)
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, cls=CustomJSONEncoder)
            
            print(f"\nResults saved to {args.output}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")
            
    print("\nDone.")

if __name__ == "__main__":
    main()
