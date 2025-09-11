#!/usr/bin/env python3
"""
Semantic compression of similar nodes in the AI Safety Intervention graph.

This script:
1. Processes cluster data prepared by auto_clustering.py
2. Uses LLMs to analyze and suggest merges between similar nodes
3. Validates and applies suggested merges to the graph database

Usage:
    python semantic_compression.py [--input-dir=clusters] [--apply-merges=False]
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import os
import numpy as np
import openai
from tqdm import tqdm
from dotenv import load_dotenv

from intervention_graph_creation.src.local_graph_extraction.db.similarity_search import SimilaritySearch
from config import load_settings

# First try to load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If not found in .env, try loading from settings
if not OPENAI_API_KEY:
    try:
        SETTINGS = load_settings()
        OPENAI_API_KEY = SETTINGS.openai.api_key
    except Exception as e:
        print(f"Warning: Could not load OpenAI API key from settings: {e}")

# Check if we have an API key
if not OPENAI_API_KEY:
    print("Warning: No OpenAI API key found in .env or settings")
else:
    openai.api_key = OPENAI_API_KEY

def load_prompt_data(file_path: str) -> Dict[str, Any]:
    """
    Load LLM prompt data from a JSON file
    
    Args:
        file_path: Path to the prompt JSON file
        
    Returns:
        Prompt data dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_llm_prompt(prompt_data: Dict[str, Any]) -> str:
    """
    Create a prompt for the LLM based on the cluster data
    
    Args:
        prompt_data: Cluster data for LLM analysis
        
    Returns:
        Formatted prompt string
    """
    nodes = prompt_data["nodes"]
    
    # Start with a system instruction
    prompt = """
You are an AI safety expert analyzing potential duplicate or overlapping concepts in an AI safety intervention knowledge graph. 
Your task is to identify whether the following concepts should be merged, and if so, how.

For each suggested merge:
1. Identify which node should be kept as the primary node and which should be merged into it
2. Explain your reasoning with specific evidence from the descriptions and relationships
3. Suggest a revised description that combines the best information from both nodes

If the nodes represent distinct concepts that should NOT be merged, explain why they are different enough to keep separate.

Here are the potentially similar nodes:
"""
    
    # Add information about each node
    for i, node in enumerate(nodes):
        prompt += f"\n--- NODE {i+1}: {node['name']} ---\n"
        prompt += f"Type: {node['type']}\n"
        prompt += f"Description: {node['description']}\n"
        
        # Add literature if available
        if node['literature']:
            prompt += f"Literature: {', '.join(node['literature'][:3])}"
            if len(node['literature']) > 3:
                prompt += f" and {len(node['literature']) - 3} more"
            prompt += "\n"
        
        # Add related nodes if available
        if node['related_nodes']:
            prompt += "Related to:\n"
            for related in node['related_nodes'][:5]:
                prompt += f"  - {related['name']} ({related['relation']})\n"
            if len(node['related_nodes']) > 5:
                prompt += f"  - And {len(node['related_nodes']) - 5} more relationships\n"
    
    # Add the final instruction
    prompt += """
\nBased on the information above, analyze these nodes and provide your recommendations:

1. Should any of these nodes be merged? If yes, which ones specifically?
2. Which node should be kept as the primary node, and which should be merged into it?
3. Why do you recommend this merge? Provide specific evidence.
4. Suggested revised description that combines information from all merged nodes.

FORMAT YOUR RESPONSE AS JSON with the following structure:
{
  "merge_recommended": true/false,
  "primary_node": "name of node to keep",
  "nodes_to_merge": ["list of node names to merge into primary"],
  "reasoning": "Your detailed explanation of why these should be merged",
  "revised_description": "Combined description for the merged node"
}

If no merge is recommended, set merge_recommended to false and explain why in the reasoning field.
"""
    
    return prompt.strip()

def query_llm(prompt: str) -> Dict[str, Any]:
    """
    Query the LLM to get merge recommendations
    
    Args:
        prompt: The formatted prompt
        
    Returns:
        Parsed response from the LLM as a dictionary
    """
    if not OPENAI_API_KEY:
        print("Error: No OpenAI API key available. Please set OPENAI_API_KEY in .env file.")
        return {
            "merge_recommended": False,
            "primary_node": None,
            "nodes_to_merge": [],
            "reasoning": "No OpenAI API key available",
            "revised_description": ""
        }
        
    try:
        # Make API call to OpenAI - handling both old and new API versions
        try:
            # Try new API client (OpenAI Python v1.0.0+)
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Get model name from environment variable or use default
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
            print(f"Using OpenAI model: {model_name}")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an AI safety expert assistant helping to improve an intervention knowledge graph."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=1000
            )
            # Extract the response text
            response_text = response.choices[0].message.content
            
        except (ImportError, AttributeError):
            # Fall back to legacy API
            print("Using legacy OpenAI API client")
            
            # Get model name from environment variable or use default
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
            print(f"Using OpenAI model: {model_name}")
            
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an AI safety expert assistant helping to improve an intervention knowledge graph."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=1000
            )
            # Extract the response text
            response_text = response.choices[0].message.content
        
        # Parse the JSON response
        # First, find JSON content (which might be within markdown code blocks)
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find any JSON-like structure
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Fall back to the entire response
                json_str = response_text
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Failed to parse JSON from LLM response.")
            print("Response:", response_text)
            return {
                "merge_recommended": False,
                "primary_node": None,
                "nodes_to_merge": [],
                "reasoning": "Failed to parse LLM response",
                "revised_description": ""
            }
            
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return {
            "merge_recommended": False,
            "primary_node": None,
            "nodes_to_merge": [],
            "reasoning": f"Error: {str(e)}",
            "revised_description": ""
        }

def validate_merge_recommendation(recommendation: Dict[str, Any], prompt_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate the merge recommendation to ensure it's safe to apply
    
    Args:
        recommendation: The merge recommendation from the LLM
        prompt_data: The original prompt data
        
    Returns:
        Tuple of (is_valid, reason)
    """
    # Check if merge is recommended
    if not recommendation.get("merge_recommended", False):
        return True, "No merge recommended"
    
    # Get node names from the prompt data
    node_names = [node["name"] for node in prompt_data["nodes"]]
    
    # Check if the primary node exists in the original nodes
    primary_node = recommendation.get("primary_node")
    if not primary_node or primary_node not in node_names:
        return False, f"Primary node '{primary_node}' not found in original nodes"
    
    # Check if nodes to merge exist in the original nodes
    nodes_to_merge = recommendation.get("nodes_to_merge", [])
    for node_name in nodes_to_merge:
        if node_name not in node_names:
            return False, f"Node to merge '{node_name}' not found in original nodes"
    
    # Check if any node is both primary and to be merged
    if primary_node in nodes_to_merge:
        return False, f"Primary node '{primary_node}' cannot be in nodes_to_merge"
    
    # Check if we have a revised description
    revised_description = recommendation.get("revised_description")
    if not revised_description or len(revised_description.strip()) < 10:
        return False, "Revised description is missing or too short"
    
    return True, "Recommendation is valid"

def apply_merge(primary_node: str, nodes_to_merge: List[str], revised_description: str) -> bool:
    """
    Apply a merge recommendation to the graph database
    
    Args:
        primary_node: Name of the node to keep
        nodes_to_merge: List of node names to merge into the primary node
        revised_description: Revised description for the merged node
        
    Returns:
        True if the merge was successful, False otherwise
    """
    try:
        # Initialize similarity search for database access
        search = SimilaritySearch()
        
        # First update the primary node's description
        if revised_description:
            description_updated = search.update_node_description(primary_node, revised_description)
            if not description_updated:
                print(f"Warning: Failed to update description for node '{primary_node}'")
                # Continue with the merge anyway
        
        # Merge each node into the primary node
        for node_to_merge in nodes_to_merge:
            success = search.execute_merge(primary_node, node_to_merge)
            if not success:
                print(f"Failed to merge '{node_to_merge}' into '{primary_node}'")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error applying merge: {e}")
        return False

def process_cluster_prompt(prompt_file: str, apply_merges: bool = False) -> Dict[str, Any]:
    """
    Process a single cluster prompt file
    
    Args:
        prompt_file: Path to the prompt JSON file
        apply_merges: Whether to apply the suggested merges to the database
        
    Returns:
        Dictionary with the results
    """
    print(f"Processing {prompt_file}...")
    
    # Load prompt data
    prompt_data = load_prompt_data(prompt_file)
    
    # Create LLM prompt
    prompt = create_llm_prompt(prompt_data)
    
    # Query LLM for merge recommendations
    recommendation = query_llm(prompt)
    
    # Validate the recommendation
    is_valid, validation_reason = validate_merge_recommendation(recommendation, prompt_data)
    
    # Apply the merge if valid and requested
    merge_applied = False
    if is_valid and recommendation.get("merge_recommended", False) and apply_merges:
        primary_node = recommendation.get("primary_node")
        nodes_to_merge = recommendation.get("nodes_to_merge", [])
        revised_description = recommendation.get("revised_description", "")
        
        merge_applied = apply_merge(primary_node, nodes_to_merge, revised_description)
    
    # Create result summary
    result = {
        "prompt_file": prompt_file,
        "cluster_id": prompt_data.get("cluster_id", "unknown"),
        "recommendation": recommendation,
        "validation": {
            "is_valid": is_valid,
            "reason": validation_reason
        },
        "merge_applied": merge_applied
    }
    
    # Save the result
    result_file = prompt_file.replace("_prompt.json", "_result.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    # Print summary
    if recommendation.get("merge_recommended", False):
        primary = recommendation.get("primary_node", "Unknown")
        to_merge = ", ".join(recommendation.get("nodes_to_merge", []))
        print(f"  Merge recommended: Keep '{primary}' and merge '{to_merge}'")
        print(f"  Validation: {'Valid' if is_valid else 'Invalid'} - {validation_reason}")
        if apply_merges:
            print(f"  Merge applied: {'Yes' if merge_applied else 'No'}")
    else:
        print(f"  No merge recommended: {recommendation.get('reasoning', 'No reason provided')}")
    
    return result

def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Perform semantic compression on similar node clusters")
    parser.add_argument("--input-dir", type=str, default="clusters",
                        help="Directory containing cluster prompt files")
    parser.add_argument("--apply-merges", action="store_true",
                        help="Apply suggested merges to the database")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model to use (default: gpt-3.5-turbo)")
    
    args = parser.parse_args()
    
    # Set the model name in the environment
    os.environ["OPENAI_MODEL_NAME"] = args.model
    print(f"Using OpenAI model: {args.model}")
    
    # Find all prompt files
    prompt_pattern = os.path.join(args.input_dir, "*_prompt.json")
    prompt_files = glob.glob(prompt_pattern)
    
    if not prompt_files:
        print(f"No prompt files found in {args.input_dir}")
        print("Run auto_clustering.py first to generate cluster data.")
        return
    
    print(f"Found {len(prompt_files)} prompt files to process")
    
    # Process each prompt file
    results = []
    for prompt_file in tqdm(prompt_files):
        result = process_cluster_prompt(prompt_file, args.apply_merges)
        results.append(result)
    
    # Summarize results
    valid_recommendations = [r for r in results if r["validation"]["is_valid"] and r["recommendation"].get("merge_recommended", False)]
    applied_merges = [r for r in valid_recommendations if r["merge_applied"]]
    
    print("\n=== Summary ===")
    print(f"Total clusters processed: {len(results)}")
    print(f"Valid merge recommendations: {len(valid_recommendations)}")
    if args.apply_merges:
        print(f"Merges successfully applied: {len(applied_merges)}")
    
    # Save overall summary
    summary = {
        "total_clusters": len(results),
        "valid_recommendations": len(valid_recommendations),
        "applied_merges": len(applied_merges) if args.apply_merges else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = os.path.join(args.input_dir, "compression_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
