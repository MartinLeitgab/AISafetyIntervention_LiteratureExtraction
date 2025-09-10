#!/usr/bin/env python
"""
Script to demonstrate using the similarity search functionality
with the FalkorDB graph database.

Usage:
    python run_similarity_search.py [--port PORT] [--query QUERY] [--threshold THRESHOLD]
"""

import argparse

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
    parser.add_argument("--mode", choices=["search", "clusters", "merges"], 
                        default="search", help="Operation mode")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Connecting to FalkorDB on port {args.port}")
    search = SimilaritySearch(port=args.port)
    
    if args.mode == "search":
        try:
            print(f"\nSearching for nodes similar to: '{args.query}'")
            similar_nodes = search.find_similar_nodes(
                text=args.query, 
                top_k=args.top_k, 
                threshold=args.threshold
            )
            
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
            print(f"\nFinding clusters with similarity threshold {args.threshold}...")
            clusters = search.find_clusters(threshold=args.threshold, min_cluster_size=2)
            
            if clusters:
                print(f"Found {len(clusters)} clusters:")
                for i, cluster in enumerate(clusters):
                    print(f"\nCluster {i+1} with {len(cluster)} nodes:")
                    for j, node in enumerate(cluster):
                        print(f"  {j+1}. {node['name']} (Similarity: {node['similarity']:.3f})")
            else:
                print("No clusters found. Try lowering the threshold or adding more nodes to the database.")
        except Exception as e:
            print(f"Error finding clusters: {e}")
            print("This could be due to FalkorDB connection issues or missing vector search capabilities.")
            print("Try restarting Docker and ensure FalkorDB is properly configured.")
    
    elif args.mode == "merges":
        try:
            print(f"\nSuggesting nodes to merge with threshold {args.threshold}...")
            merges = search.suggest_merges(threshold=args.threshold)
            
            if merges:
                print(f"Found {len(merges)} potential merges:")
                for i, merge in enumerate(merges):
                    primary = merge['primary_node']
                    candidates = merge['merge_candidates']
                    print(f"\nMerge {i+1}: Keep '{primary['name']}' and merge:")
                    for j, candidate in enumerate(candidates):
                        print(f"  {j+1}. '{candidate['name']}' (Similarity: {candidate['similarity']:.3f})")
                    
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

if __name__ == "__main__":
    main()
