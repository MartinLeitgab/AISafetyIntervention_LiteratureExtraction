#!/usr/bin/env python
"""
Script for LLM-based semantic compression of graph node clusters.

This script takes the output from auto_clustering.py and uses an LLM to:
1. Analyze clusters of similar nodes
2. Create semantically compressed representations
3. Generate new merged nodes that preserve the essential information
4. Prepare data for updating the graph database

Usage:
    python semantic_compression.py [options]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import time
import glob

# Try importing LLM libraries, provide helpful error if missing
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not found. Install with: pip install openai")

# Import project modules
from intervention_graph_creation.src.local_graph_extraction.db.similarity_search import SimilaritySearch

# Semantic compression prompt template
COMPRESSION_PROMPT_TEMPLATE = """
# Semantic Compression of Similar AI Safety Intervention Concepts

## Task Description
You are given a cluster of similar nodes from a knowledge graph about AI safety interventions. 
Each node represents a concept, finding, or intervention in AI safety research.
Your task is to analyze the semantic content of these nodes and create a compressed representation that:
1. Preserves the essential information from all nodes in the cluster
2. Maintains the relationships to other nodes in the graph
3. Creates a coherent, unified concept that can replace the individual nodes

## Node Cluster Information
Primary Node: {primary_node_name}
Type: {primary_node_type}
Description: {primary_node_description}

Similar Nodes:
{similar_nodes}

## Neighborhood Information
The primary node has the following connections:
{primary_connections}

The similar nodes have these connections:
{similar_connections}

## Output Instructions
Create a semantically compressed node that represents the entire cluster. Provide:

1. A concise name for the compressed concept (max 100 characters)
2. The most appropriate node type 
3. A comprehensive description that captures all important aspects (200-500 words)
4. A list of the original node names this compressed node replaces
5. The key connections that should be preserved, with relationship types and directions
6. A brief semantic summary of how these concepts relate to each other (100-200 words)

## Output Format
Return ONLY a valid JSON object with the following structure:
```json
{
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
```
"""

def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based semantic compression of graph node clusters")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing cluster data from auto_clustering.py")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save compressed nodes (defaults to input_dir/compressed)")
    parser.add_argument("--llm-model", type=str, default="gpt-4",
                        help="LLM model to use for compression (default: gpt-4)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (can also use OPENAI_API_KEY env variable)")
    parser.add_argument("--batch-file", type=str, default=None,
                        help="Process a specific batch file instead of all batches")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls in seconds (default: 1.0)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries for failed API calls")
    parser.add_argument("--update-db", action="store_true",
                        help="Update the graph database with compressed nodes")
    parser.add_argument("--db-port", type=int, default=6379,
                        help="FalkorDB port (default: 6379)")
    
    return parser.parse_args()

def setup_output_directory(input_dir, output_dir=None):
    """Create output directory for compressed nodes"""
    input_path = Path(input_dir)
    
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = input_path / "compressed"
    
    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path

def find_batch_files(input_dir):
    """Find all LLM batch files in the input directory"""
    input_path = Path(input_dir)
    
    # Check if there's a llm_batches directory
    llm_batches_dir = input_path / "llm_batches"
    if llm_batches_dir.exists() and llm_batches_dir.is_dir():
        batch_files = list(llm_batches_dir.glob("batch_*.json"))
        if batch_files:
            return batch_files
    
    # If not, look for merge_suggestions.json
    merge_suggestions_file = input_path / "merge_suggestions.json"
    if merge_suggestions_file.exists():
        return [merge_suggestions_file]
    
    # If nothing found, look for any JSON files
    json_files = list(input_path.glob("*.json"))
    if json_files:
        return json_files
    
    return []

def format_node_data_for_prompt(node_data):
    """Format node data for inclusion in the prompt"""
    node_info = f"Name: {node_data['name']}\n"
    
    if 'type' in node_data and node_data['type']:
        node_info += f"Type: {node_data['type']}\n"
    
    if 'description' in node_data and node_data['description']:
        desc = node_data['description']
        if len(desc) > 300:
            desc = desc[:297] + "..."
        node_info += f"Description: {desc}\n"
    
    return node_info

def format_connections_for_prompt(neighborhood):
    """Format neighborhood connections for inclusion in the prompt"""
    if not neighborhood or 'edges' not in neighborhood:
        return "No connections found."
    
    edges = neighborhood.get('edges', [])
    if not edges:
        return "No connections found."
    
    formatted_connections = []
    
    # Group edges by target node
    connections_by_target = {}
    for edge in edges:
        target = edge.get('target', 'Unknown')
        rel_type = edge.get('type', 'RELATED_TO')
        
        if target not in connections_by_target:
            connections_by_target[target] = []
        
        connections_by_target[target].append(rel_type)
    
    # Format connections
    for target, rel_types in connections_by_target.items():
        # Remove duplicates
        unique_rel_types = list(set(rel_types))
        rel_str = ', '.join(unique_rel_types)
        
        formatted_connections.append(f"- Connected to '{target}' via: {rel_str}")
    
    return "\n".join(formatted_connections)

def create_compression_prompt(cluster_data):
    """Create a prompt for LLM-based semantic compression of a node cluster"""
    primary_node = cluster_data.get('primary_node', {})
    candidates = cluster_data.get('merge_candidates', [])
    
    # Format primary node information
    primary_name = primary_node.get('name', 'Unknown')
    primary_type = primary_node.get('type', 'Unknown')
    primary_desc = primary_node.get('description', 'No description available')
    
    # Format similar nodes information
    similar_nodes_text = ""
    for i, candidate in enumerate(candidates):
        similar_nodes_text += f"Node {i+1}:\n"
        similar_nodes_text += format_node_data_for_prompt(candidate)
        similar_nodes_text += f"Similarity to primary: {candidate.get('similarity', 0):.4f}\n\n"
    
    # Format neighborhood information
    primary_neighborhood = primary_node.get('neighborhood', {})
    primary_connections = format_connections_for_prompt(primary_neighborhood)
    
    # Format connections for similar nodes
    similar_connections_text = ""
    for i, candidate in enumerate(candidates):
        candidate_neighborhood = candidate.get('neighborhood', {})
        connections = format_connections_for_prompt(candidate_neighborhood)
        
        similar_connections_text += f"Node '{candidate.get('name', 'Unknown')}' connections:\n"
        similar_connections_text += connections + "\n\n"
    
    # Create the full prompt
    prompt = COMPRESSION_PROMPT_TEMPLATE.format(
        primary_node_name=primary_name,
        primary_node_type=primary_type,
        primary_node_description=primary_desc,
        similar_nodes=similar_nodes_text,
        primary_connections=primary_connections,
        similar_connections=similar_connections_text
    )
    
    return prompt

def call_openai_api(prompt, model="gpt-4", api_key=None, max_retries=3, delay=1.0):
    """Call OpenAI API to generate a compressed node representation"""
    if not OPENAI_AVAILABLE:
        print("Error: OpenAI library not installed. Run 'pip install openai'")
        return None
    
    # Use provided API key or environment variable
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: No OpenAI API key provided")
        return None
    
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an AI safety research assistant specializing in knowledge graph compression."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic output
                max_tokens=1500,  # Adjust based on expected output length
                n=1,
                stop=None
            )
            
            # Extract the content from the response
            content = response.choices[0].message.content
            
            # Extract JSON from content (it might be wrapped in markdown code blocks)
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # If not in code blocks, use the whole response
                json_str = content
            
            # Parse JSON
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                print("Error: Could not parse JSON from API response")
                print("Raw content:", content)
                return {"error": "JSON parsing failed", "raw_content": content}
            
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Failed to get API response.")
                return {"error": str(e)}
    
    return None

def process_batch_file(batch_file_path, output_dir, args):
    """Process a single batch file of clusters"""
    print(f"Processing batch file: {batch_file_path}")
    
    try:
        with open(batch_file_path, 'r') as f:
            batch_data = json.load(f)
        
        # Check if it's a standard batch file or merge suggestions
        clusters = []
        
        if 'clusters' in batch_data:
            # Standard batch format from auto_clustering.py
            clusters = batch_data['clusters']
        elif isinstance(batch_data, list):
            # Direct merge suggestions format
            clusters = batch_data
        else:
            print(f"Error: Unrecognized data format in {batch_file_path}")
            return None
        
        compressed_results = []
        
        for i, cluster_data in enumerate(clusters):
            print(f"Processing cluster {i+1}/{len(clusters)}")
            
            # Create compression prompt
            prompt = create_compression_prompt(cluster_data)
            
            # Save the prompt for reference
            prompt_file = output_dir / f"prompt_cluster_{i+1}.txt"
            with open(prompt_file, 'w') as f:
                f.write(prompt)
            
            # Call LLM API if available
            if OPENAI_AVAILABLE and args.api_key:
                print("Calling OpenAI API...")
                result = call_openai_api(
                    prompt=prompt,
                    model=args.llm_model,
                    api_key=args.api_key,
                    max_retries=args.max_retries,
                    delay=args.delay
                )
                
                if result:
                    # Add cluster information to result
                    if 'compressed_node' in result:
                        result['cluster_info'] = {
                            'primary_node': cluster_data.get('primary_node', {}).get('name', 'Unknown'),
                            'merge_candidates': [c.get('name', 'Unknown') for c in cluster_data.get('merge_candidates', [])]
                        }
                    
                    compressed_results.append(result)
                    
                    # Save individual result
                    result_file = output_dir / f"compressed_cluster_{i+1}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"Saved compression result to {result_file}")
                    
                    # Add a delay between API calls
                    if i < len(clusters) - 1:
                        time.sleep(args.delay)
                else:
                    print("Error: No result from API call")
            else:
                print("Skipping API call (OpenAI library not available or no API key provided)")
                print(f"Prompt saved to {prompt_file}")
                print("You can manually process this prompt with an LLM and save the result")
        
        # Save all results in a single file
        all_results_file = output_dir / f"all_compressed_results_{Path(batch_file_path).stem}.json"
        with open(all_results_file, 'w') as f:
            json.dump(compressed_results, f, indent=2)
        
        print(f"\nAll compression results saved to {all_results_file}")
        return compressed_results
    
    except Exception as e:
        print(f"Error processing batch file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def update_graph_database(compressed_results, db_port):
    """Update the graph database with compressed nodes"""
    print("\nUpdating graph database with compressed nodes...")
    
    # Initialize similarity search (which has database access)
    search = SimilaritySearch(port=db_port)
    
    updated_nodes = []
    failed_updates = []
    
    for result in compressed_results:
        if 'compressed_node' not in result:
            print("Error: Missing compressed_node in result")
            continue
        
        compressed = result['compressed_node']
        cluster_info = result.get('cluster_info', {})
        
        # Get the merged nodes
        primary_node = cluster_info.get('primary_node', compressed.get('merged_from', [])[0] if compressed.get('merged_from') else None)
        merge_candidates = compressed.get('merged_from', [])
        
        if not primary_node:
            print("Error: No primary node identified for compression result")
            failed_updates.append({
                "compressed_node": compressed,
                "error": "No primary node identified"
            })
            continue
        
        try:
            # Create or update the compressed node
            # Note: This depends on how your graph database handles node creation/updates
            # Here we're using the execute_merge method from SimilaritySearch
            
            # First, remove the primary node from the merge candidates if it's there
            if primary_node in merge_candidates:
                merge_candidates.remove(primary_node)
            
            # Track successful merges
            successful_merges = []
            
            # Merge each candidate into the primary node
            for candidate in merge_candidates:
                try:
                    success = search.execute_merge(primary_node, candidate)
                    if success:
                        print(f"  ✓ Merged '{candidate}' into '{primary_node}'")
                        successful_merges.append(candidate)
                    else:
                        print(f"  ✗ Failed to merge '{candidate}' into '{primary_node}'")
                except Exception as e:
                    print(f"  ✗ Error merging '{candidate}' into '{primary_node}': {e}")
            
            # Update the primary node with the compressed information
            # This depends on your database schema and update methods
            # As a placeholder, we'll just print what would be updated
            print(f"\nWould update node '{primary_node}' with:")
            print(f"  Name: {compressed.get('name')}")
            print(f"  Type: {compressed.get('type')}")
            print(f"  Description: {compressed.get('description')[:100]}..." if len(compressed.get('description', '')) > 100 else f"  Description: {compressed.get('description')}")
            
            updated_nodes.append({
                "original_primary": primary_node,
                "merged_nodes": successful_merges,
                "compressed_node": compressed
            })
            
        except Exception as e:
            print(f"Error updating database with compressed node: {str(e)}")
            failed_updates.append({
                "compressed_node": compressed,
                "error": str(e)
            })
    
    # Save update results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_nodes": updated_nodes,
        "failed_updates": failed_updates
    }
    
    return results

def main():
    args = parse_args()
    
    # Set up output directory
    output_dir = setup_output_directory(args.input_dir, args.output_dir)
    
    # Find batch files to process
    if args.batch_file:
        batch_files = [Path(args.batch_file)]
        if not batch_files[0].exists():
            print(f"Error: Specified batch file {args.batch_file} not found")
            return
    else:
        batch_files = find_batch_files(args.input_dir)
    
    if not batch_files:
        print("Error: No batch files found")
        return
    
    print(f"Found {len(batch_files)} batch files to process")
    
    # Check OpenAI API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No OpenAI API key provided. Will save prompts only.")
    elif not OPENAI_AVAILABLE:
        print("Warning: OpenAI library not installed. Will save prompts only.")
    
    all_compressed_results = []
    
    # Process each batch file
    for batch_file in batch_files:
        results = process_batch_file(batch_file, output_dir, args)
        if results:
            all_compressed_results.extend(results)
    
    # Update graph database if requested
    if args.update_db and all_compressed_results:
        update_results = update_graph_database(all_compressed_results, args.db_port)
        
        # Save update results
        update_results_file = output_dir / "db_update_results.json"
        with open(update_results_file, 'w') as f:
            json.dump(update_results, f, indent=2)
        
        print(f"\nDatabase update results saved to {update_results_file}")
    
    print("\nSemantic compression process complete!")

if __name__ == "__main__":
    main()
