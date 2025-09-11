#!/usr/bin/env python3
"""
Automated clustering and semantic compression of the AI Safety Intervention graph.

This script:
1. Finds clusters of similar nodes in the graph using vector similarity
2. Creates a JSON structure for each cluster with node details and relationships
3. Prepares data for semantic compression with LLMs

Usage:
    python auto_clustering.py [--threshold=0.85] [--cluster-size=5] [--output-dir=clusters]
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from intervention_graph_creation.src.local_graph_extraction.db.similarity_search import SimilaritySearch


def format_cluster_data(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format cluster data for human review and LLM processing
    
    Args:
        cluster: List of nodes in the cluster with their properties
        
    Returns:
        Formatted cluster data
    """
    # Format cluster data with a unique ID and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cluster_id = f"cluster_{timestamp}"
    
    # Extract node names for the summary
    node_names = [node.get("name", "Unnamed Node") for node in cluster]
    
    # Create the cluster data structure
    cluster_data = {
        "cluster_id": cluster_id,
        "timestamp": timestamp,
        "node_count": len(cluster),
        "nodes": node_names,
        "detailed_nodes": []
    }
    
    # Add detailed information for each node
    for node in cluster:
        node_name = node.get("name", "Unnamed Node")
        node_type = node.get("type", "")
        node_description = node.get("description", "")
        similarity = node.get("similarity", 0.0)
        
        # Get neighborhood information if available
        neighborhood = node.get("neighborhood", {})
        edges = neighborhood.get("edges", [])
        neighbors = neighborhood.get("neighbors", {})
        
        # Format edge information
        formatted_edges = []
        for edge in edges:
            edge_type = edge.get("type", "RELATED_TO")
            target_name = edge.get("target", "Unknown")
            edge_props = edge.get("properties", {})
            
            # Get target node details from neighbors dictionary
            target_info = neighbors.get(target_name, {})
            target_labels = target_info.get("labels", ["NODE"])
            target_props = target_info.get("properties", {})
            
            formatted_edges.append({
                "type": edge_type,
                "target": target_name,
                "target_type": target_props.get("type", ""),
                "target_description": target_props.get("description", ""),
                "properties": edge_props
            })
        
        # Add literature connections if available
        literature = node.get("literature", [])
        
        # Add the node with its full details
        cluster_data["detailed_nodes"].append({
            "name": node_name,
            "type": node_type,
            "description": node_description,
            "similarity": similarity,
            "edges": formatted_edges,
            "literature": literature
        })
    
    return cluster_data

def save_cluster_data(cluster_data: Dict[str, Any], output_dir: str) -> str:
    """
    Save cluster data to a JSON file
    
    Args:
        cluster_data: Formatted cluster data
        output_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from cluster ID
    cluster_id = cluster_data["cluster_id"]
    filename = f"{cluster_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, indent=2)
    
    return filepath

def prepare_llm_prompt(cluster_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare data for semantic compression with LLMs
    
    Args:
        cluster_data: Formatted cluster data
        
    Returns:
        Data structure for LLM semantic compression
    """
    node_names = [node["name"] for node in cluster_data["detailed_nodes"]]
    
    # Create a simplified version for the prompt
    prompt_data = {
        "task": "semantic_compression",
        "cluster_id": cluster_data["cluster_id"],
        "nodes": [],
        "request": f"Analyze these {len(node_names)} potentially similar AI safety intervention concepts and suggest merges."
    }
    
    # Add detailed node information for the prompt
    for node in cluster_data["detailed_nodes"]:
        # Get literature details for evidence
        literature = [lit.get("title", "") for lit in node.get("literature", [])]
        
        # Get related nodes
        related_nodes = []
        for edge in node.get("edges", []):
            related_nodes.append({
                "name": edge.get("target", ""),
                "type": edge.get("target_type", ""),
                "relation": edge.get("type", "RELATED_TO")
            })
        
        # Add the node to the prompt data
        prompt_data["nodes"].append({
            "name": node.get("name", ""),
            "type": node.get("type", ""),
            "description": node.get("description", ""),
            "similarity": node.get("similarity", 0.0),
            "literature": literature,
            "related_nodes": related_nodes
        })
    
    return prompt_data

def process_clusters(threshold: float = 0.85, cluster_size: int = 5, output_dir: str = "clusters") -> List[str]:
    """
    Process clusters and prepare them for review and semantic compression
    
    Args:
        threshold: Similarity threshold (0-1)
        cluster_size: Maximum cluster size
        output_dir: Directory to save cluster data
        
    Returns:
        List of paths to saved cluster files
    """
    # Initialize similarity search
    search = SimilaritySearch()
    
    # Find clusters with neighborhood information
    print(f"Finding clusters with similarity threshold {threshold} and max size {cluster_size}...")
    clusters = search.get_cluster_with_neighborhoods(threshold=threshold, max_cluster_size=cluster_size)
    
    if not clusters:
        print("No clusters found. Try lowering the threshold.")
        return []
    
    print(f"Found {len(clusters)} clusters")
    
    # Process each cluster
    saved_files = []
    for i, cluster in enumerate(clusters):
        print(f"Processing cluster {i+1} with {len(cluster)} nodes...")
        
        # Format and save cluster data
        cluster_data = format_cluster_data(cluster)
        filepath = save_cluster_data(cluster_data, output_dir)
        saved_files.append(filepath)
        
        # Prepare LLM prompt
        prompt_data = prepare_llm_prompt(cluster_data)
        prompt_filepath = filepath.replace(".json", "_prompt.json")
        with open(prompt_filepath, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=2)
        
        saved_files.append(prompt_filepath)
        
        print(f"  Saved cluster data to {filepath}")
        print(f"  Saved LLM prompt to {prompt_filepath}")
    
    return saved_files

def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Find clusters of similar nodes and prepare for semantic compression")
    parser.add_argument("--threshold", type=float, default=0.85, 
                        help="Similarity threshold (0-1)")
    parser.add_argument("--cluster-size", type=int, default=5, 
                        help="Maximum cluster size")
    parser.add_argument("--output-dir", type=str, default="clusters",
                        help="Directory to save cluster data")
    
    args = parser.parse_args()
    
    # Run the cluster processing
    saved_files = process_clusters(
        threshold=args.threshold,
        cluster_size=args.cluster_size,
        output_dir=args.output_dir
    )
    
    if saved_files:
        print(f"\nSuccessfully processed {len(saved_files)//2} clusters")
        print(f"Saved data to {args.output_dir} directory")
    else:
        print("No clusters were processed. Try adjusting the threshold or cluster size.")

if __name__ == "__main__":
    main()
