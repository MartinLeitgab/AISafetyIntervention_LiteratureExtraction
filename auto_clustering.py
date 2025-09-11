#!/usr/bin/env python
"""
Script to automate the process of finding and analyzing clusters of similar nodes
in the FalkorDB graph database. This script is part of the workflow for semantic compression
and node merging in the AI Safety Intervention Literature Extraction project.

Workflow steps:
1. Find clusters of similar nodes based on cosine similarity
2. For each cluster, collect edge and neighbor information
3. Generate merge suggestions with detailed neighborhood data
4. (Optional) Execute merges or export data for LLM-based semantic compression

Usage:
    python auto_clustering.py [options]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from intervention_graph_creation.src.local_graph_extraction.db.similarity_search import SimilaritySearch

def parse_args():
    parser = argparse.ArgumentParser(description="Automate node clustering and merge suggestions")
    parser.add_argument("--port", type=int, default=6379, 
                        help="FalkorDB port (default: 6379)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Similarity threshold (0-1)")
    parser.add_argument("--max-cluster-size", type=int, default=10,
                        help="Maximum number of nodes in a cluster")
    parser.add_argument("--min-cluster-size", type=int, default=2,
                        help="Minimum number of nodes in a cluster")
    parser.add_argument("--output-dir", type=str, default="cluster_data",
                        help="Directory to save cluster data (default: cluster_data)")
    parser.add_argument("--auto-merge", action="store_true",
                        help="Automatically merge nodes with very high similarity (>0.9)")
    parser.add_argument("--export-for-llm", action="store_true",
                        help="Export cluster data for LLM-based semantic compression")
    parser.add_argument("--llm-batch-size", type=int, default=5,
                        help="Number of clusters to include in each LLM batch")
    
    return parser.parse_args()

def create_output_directory(base_dir):
    """Create a timestamped output directory for cluster data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"clusters_{timestamp}"
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def save_cluster_data(output_dir, clusters, enriched=False):
    """Save cluster data to JSON files"""
    if enriched:
        filename = output_dir / "enriched_clusters.json"
    else:
        filename = output_dir / "basic_clusters.json"
    
    # Create a custom JSON encoder to handle non-serializable types
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            try:
                return json.JSONEncoder.default(self, obj)
            except TypeError:
                return str(obj)
    
    with open(filename, 'w') as f:
        json.dump(clusters, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"Saved cluster data to {filename}")
    return filename

def save_merge_suggestions(output_dir, merges):
    """Save merge suggestions to a JSON file"""
    filename = output_dir / "merge_suggestions.json"
    
    # Create a custom JSON encoder to handle non-serializable types
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            try:
                return json.JSONEncoder.default(self, obj)
            except TypeError:
                return str(obj)
    
    with open(filename, 'w') as f:
        json.dump(merges, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"Saved merge suggestions to {filename}")
    return filename

def create_llm_input_batches(output_dir, merges, batch_size=5):
    """
    Create input files for LLM-based semantic compression
    Each batch will contain data for batch_size clusters
    """
    batches_dir = output_dir / "llm_batches"
    batches_dir.mkdir(exist_ok=True)
    
    # Split merge suggestions into batches
    total_batches = (len(merges) + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(merges))
        batch_merges = merges[start_idx:end_idx]
        
        batch_filename = batches_dir / f"batch_{batch_idx + 1}_of_{total_batches}.json"
        
        # Create the LLM input format
        llm_input = {
            "batch_info": {
                "batch_number": batch_idx + 1,
                "total_batches": total_batches,
                "clusters_in_batch": len(batch_merges)
            },
            "instructions": (
                "For each cluster, analyze the semantic content of all nodes and their connections. "
                "Create a compressed representation that preserves the key information. "
                "Generate a new node that can replace the cluster nodes while maintaining their "
                "connections to neighboring nodes. Return the compressed node in the specified format."
            ),
            "output_format": {
                "compressed_nodes": [
                    {
                        "cluster_id": "integer",
                        "compressed_node": {
                            "name": "string - concise name for the merged concept",
                            "type": "string - most appropriate node type",
                            "description": "string - comprehensive description capturing all important aspects",
                            "merged_from": ["list of original node names"],
                            "key_connections": [
                                {
                                    "target": "string - target node name",
                                    "relationship": "string - relationship type",
                                    "direction": "string - 'incoming' or 'outgoing'"
                                }
                            ],
                            "semantic_summary": "string - summary of the semantic content"
                        }
                    }
                ]
            },
            "clusters": []
        }
        
        # Add each merge suggestion to the batch
        for idx, merge in enumerate(batch_merges):
            cluster_data = {
                "cluster_id": start_idx + idx + 1,
                "primary_node": merge["primary_node"],
                "merge_candidates": merge["merge_candidates"]
            }
            llm_input["clusters"].append(cluster_data)
        
        # Save the batch file
        with open(batch_filename, 'w') as f:
            json.dump(llm_input, f, indent=2)
        
        print(f"Created LLM input batch: {batch_filename}")
    
    return batches_dir

def execute_auto_merges(search, merges, threshold=0.9):
    """Execute merges automatically for nodes with very high similarity"""
    auto_merged = []
    
    print(f"\nExecuting automatic merges for nodes with similarity > {threshold}...")
    
    for merge_idx, merge in enumerate(merges):
        primary = merge['primary_node']
        candidates_to_merge = []
        
        # Find candidates with similarity above threshold
        for candidate in merge['merge_candidates']:
            if candidate['similarity'] >= threshold:
                candidates_to_merge.append(candidate)
        
        if candidates_to_merge:
            print(f"\nMerge {merge_idx+1}: Keeping '{primary['name']}' and merging:")
            
            for candidate in candidates_to_merge:
                try:
                    success = search.execute_merge(primary['name'], candidate['name'])
                    if success:
                        print(f"  ✓ Merged '{candidate['name']}' into '{primary['name']}'")
                        auto_merged.append({
                            "primary": primary['name'],
                            "merged": candidate['name'],
                            "similarity": candidate['similarity'],
                            "success": True
                        })
                    else:
                        print(f"  ✗ Failed to merge '{candidate['name']}' into '{primary['name']}'")
                        auto_merged.append({
                            "primary": primary['name'],
                            "merged": candidate['name'],
                            "similarity": candidate['similarity'],
                            "success": False
                        })
                except Exception as e:
                    print(f"  ✗ Error merging '{candidate['name']}': {e}")
                    auto_merged.append({
                        "primary": primary['name'],
                        "merged": candidate['name'],
                        "similarity": candidate['similarity'],
                        "success": False,
                        "error": str(e)
                    })
    
    return auto_merged

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    # Initialize similarity search
    print(f"Connecting to FalkorDB on port {args.port}")
    search = SimilaritySearch(port=args.port)
    
    # Step 1: Find basic clusters
    print(f"\nFinding clusters with similarity threshold {args.threshold}...")
    print(f"Cluster size limits: min {args.min_cluster_size}, max {args.max_cluster_size}")
    
    try:
        clusters = search.find_clusters(
            threshold=args.threshold,
            min_cluster_size=args.min_cluster_size,
            max_cluster_size=args.max_cluster_size
        )
        
        if not clusters:
            print("No clusters found. Try lowering the threshold or adding more nodes with embeddings.")
            return
        
        # Save basic cluster data
        basic_clusters_file = save_cluster_data(output_dir, clusters)
        
        # Print summary of clusters found
        print(f"\nFound {len(clusters)} clusters:")
        total_nodes = sum(len(cluster) for cluster in clusters)
        print(f"Total nodes in clusters: {total_nodes}")
        
        cluster_sizes = [len(cluster) for cluster in clusters]
        avg_size = sum(cluster_sizes) / len(cluster_sizes)
        print(f"Average cluster size: {avg_size:.2f}")
        print(f"Cluster size distribution: {cluster_sizes}")
        
        # Step 2: Get enriched clusters with neighborhood information
        print(f"\nCollecting neighborhood information for all clusters...")
        enriched_clusters = search.get_cluster_with_neighborhoods(
            threshold=args.threshold,
            max_cluster_size=args.max_cluster_size
        )
        
        # Save enriched cluster data
        enriched_clusters_file = save_cluster_data(output_dir, enriched_clusters, enriched=True)
        
        # Step 3: Generate merge suggestions
        print(f"\nGenerating merge suggestions...")
        merges = search.suggest_merges(
            threshold=args.threshold,
            max_cluster_size=args.max_cluster_size
        )
        
        # Save merge suggestions
        merge_suggestions_file = save_merge_suggestions(output_dir, merges)
        
        # Step 4: Execute automatic merges if requested
        auto_merge_results = None
        if args.auto_merge:
            auto_merge_results = execute_auto_merges(search, merges, threshold=0.9)
            
            # Save auto-merge results
            auto_merge_file = output_dir / "auto_merge_results.json"
            with open(auto_merge_file, 'w') as f:
                json.dump(auto_merge_results, f, indent=2)
            print(f"\nSaved auto-merge results to {auto_merge_file}")
        
        # Step 5: Create LLM input batches if requested
        if args.export_for_llm:
            print(f"\nCreating LLM input batches (batch size: {args.llm_batch_size})...")
            llm_batches_dir = create_llm_input_batches(
                output_dir, 
                merges, 
                batch_size=args.llm_batch_size
            )
            
            # Create a README for the LLM batches
            readme_file = llm_batches_dir / "README.md"
            with open(readme_file, 'w') as f:
                f.write("# LLM Semantic Compression Batches\n\n")
                f.write("These files contain node cluster data for LLM-based semantic compression.\n\n")
                f.write("## Instructions\n\n")
                f.write("1. Process each batch file with an LLM (e.g., GPT-4)\n")
                f.write("2. The LLM should analyze each cluster and generate a compressed representation\n")
                f.write("3. Save the LLM output for each batch\n")
                f.write("4. Use the output to create merged/compressed nodes in the graph database\n\n")
                f.write("## Batch Information\n\n")
                f.write(f"Total batches: {(len(merges) + args.llm_batch_size - 1) // args.llm_batch_size}\n")
                f.write(f"Total clusters: {len(merges)}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"Created README for LLM batches at {readme_file}")
        
        # Create a summary file
        summary_file = output_dir / "summary.json"
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "threshold": args.threshold,
                "min_cluster_size": args.min_cluster_size,
                "max_cluster_size": args.max_cluster_size,
                "auto_merge": args.auto_merge
            },
            "results": {
                "total_clusters": len(clusters),
                "total_nodes_in_clusters": total_nodes,
                "average_cluster_size": avg_size,
                "cluster_size_distribution": cluster_sizes,
                "auto_merges_executed": len(auto_merge_results) if auto_merge_results else 0
            },
            "files": {
                "basic_clusters": str(basic_clusters_file),
                "enriched_clusters": str(enriched_clusters_file),
                "merge_suggestions": str(merge_suggestions_file)
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nProcess complete! Summary saved to {summary_file}")
        
    except Exception as e:
        print(f"Error during clustering process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
