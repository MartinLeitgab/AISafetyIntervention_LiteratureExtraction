#!/usr/bin/env python3
"""
Direct runner for semantic compression workflow.
This script skips environment checks and runs the workflow directly.

Usage:
    python direct_run_semantic_compression.py
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

def setup_output_dir() -> str:
    """Create a timestamped output directory for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"clusters_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run_clustering(threshold: float, cluster_size: int, output_dir: str) -> bool:
    """Run the auto clustering script"""
    print("\n=== Step 1: Finding Similar Node Clusters ===")
    
    # Directly import and run the clustering function
    try:
        sys.path.append(os.path.abspath("."))
        from intervention_graph_creation.src.local_graph_extraction.db.auto_clustering import process_clusters
        
        print(f"Finding clusters with threshold={threshold}, cluster_size={cluster_size}...")
        saved_files = process_clusters(threshold=threshold, cluster_size=cluster_size, output_dir=output_dir)
        
        if not saved_files:
            print("No clusters found or processing failed.")
            return False
        
        print(f"Successfully processed {len(saved_files)//2} clusters")
        return True
    except Exception as e:
        print(f"Error running clustering: {e}")
        return False

def load_dotenv_if_available():
    """Load .env file if python-dotenv is installed"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded environment variables from .env file")
    except ImportError:
        print("python-dotenv not installed, skipping .env loading")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("Found OpenAI API key in environment variables")
    else:
        print("Warning: OPENAI_API_KEY not found in environment variables")

def run_semantic_compression(output_dir: str, apply_merges: bool, model: str = "gpt-3.5-turbo") -> bool:
    """Run semantic compression directly"""
    print(f"\n=== Step 2: Analyzing Clusters for Semantic Compression (Using {model}) ===")
    
    try:
        # First load environment variables if possible
        load_dotenv_if_available()
        
        # Import and run compression function
        sys.path.append(os.path.abspath("."))
        from intervention_graph_creation.src.local_graph_extraction.db.semantic_compression import process_cluster_prompt
        import glob
        
        # Find all prompt files
        prompt_pattern = os.path.join(output_dir, "*_prompt.json")
        prompt_files = glob.glob(prompt_pattern)
        
        if not prompt_files:
            print(f"No prompt files found in {output_dir}")
            return False
        
        print(f"Found {len(prompt_files)} prompt files to process")
        
        # Set the model name in the environment for semantic_compression.py to use
        os.environ["OPENAI_MODEL_NAME"] = model
        print(f"Set OpenAI model to use: {model}")
        
        # Process each prompt file
        successful = 0
        for prompt_file in prompt_files:
            try:
                result = process_cluster_prompt(prompt_file, apply_merges)
                if result.get("recommendation", {}).get("merge_recommended", False):
                    print(f"Merge recommended for {os.path.basename(prompt_file)}")
                    successful += 1
            except Exception as e:
                print(f"Error processing {prompt_file}: {e}")
        
        print(f"Successfully processed {successful} clusters with merge recommendations")
        return True
    except Exception as e:
        print(f"Error running semantic compression: {e}")
        return False

def main():
    """Run the workflow directly without checks"""
    parser = argparse.ArgumentParser(description="Run semantic compression workflow directly (skips environment checks)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Similarity threshold for clustering (0-1)")
    parser.add_argument("--cluster-size", type=int, default=5,
                        help="Maximum cluster size")
    parser.add_argument("--apply-merges", action="store_true",
                        help="Apply validated merges to the graph")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model to use (default: gpt-3.5-turbo)")
    
    args = parser.parse_args()
    
    print("=== Running Semantic Compression Workflow Directly ===")
    print("Note: This script bypasses environment checks")
    print(f"Parameters: threshold={args.threshold}, cluster_size={args.cluster_size}, apply_merges={args.apply_merges}")
    
    # Create output directory
    output_dir = setup_output_dir()
    print(f"Saving all results to {output_dir}")
    
    # Run clustering
    if not run_clustering(args.threshold, args.cluster_size, output_dir):
        print("Clustering failed, stopping workflow.")
        return
    
    # Run semantic compression
    if not run_semantic_compression(output_dir, args.apply_merges, args.model):
        print("Semantic compression failed.")
        return
    
    # Print summary
    print("\n=== Workflow Complete ===")
    print(f"All results saved to {output_dir}")
    if args.apply_merges:
        print("Merges were applied to the graph database.")
    else:
        print("No merges were applied. Run with --apply-merges to apply validated merges.")

if __name__ == "__main__":
    main()
