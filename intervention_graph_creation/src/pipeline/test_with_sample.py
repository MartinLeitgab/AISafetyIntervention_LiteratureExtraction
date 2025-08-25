#!/usr/bin/env python3
"""
Test script to run the pipeline with output_sample.json.

This script tests the complete pipeline with the actual sample data.
"""

import json
import tempfile
from pathlib import Path
import os

# Add the project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import Pipeline, EmbedderFlow, DatabaseDispatchFlow


def create_test_jsonl_from_sample():
    """Create a JSONL file from the output_sample.json."""
    sample_path = Path(__file__).parent.parent.parent / "src" / "prompt" / "schemas" / "output_sample.json"
    
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")
    
    # Read the sample JSON
    with open(sample_path, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    # Create temporary JSONL file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    temp_file.write(json.dumps(sample_data) + '\n')
    temp_file.close()
    
    print(f"Created test JSONL file: {temp_file.name}")
    return temp_file.name


def test_database_connection():
    """Test the database connection."""
    print("Testing database connection...")
    
    try:
        from falkordb import FalkorDB
        db = FalkorDB()
        # Try to create a test graph
        g = db.select_graph("AISafetyIntervention")
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def run_pipeline_test():
    """Run the complete pipeline test."""
    print("Starting pipeline test with sample data...")
    
    # Create test JSONL file
    jsonl_path = create_test_jsonl_from_sample()
    
    try:
        # Test database connection first
        if not test_database_connection():
            print("Skipping database operations due to connection failure")
            db_flow = None
        else:
            db_flow = DatabaseDispatchFlow(host="localhost", port=6379, graph_name="AISafetyIntervention")
        
        # Create pipeline
        pipeline = Pipeline()
        
        # Create flows
        embedder_flow = EmbedderFlow()
        
        # Add flows to pipeline
        pipeline.push_node(embedder_flow)
        if db_flow:
            pipeline.push_node(db_flow)
        
        print(f"Pipeline created with {len(pipeline._chain)} flows")
        
        # Process the JSONL file
        print(f"Processing {jsonl_path}...")
        result = pipeline.process(jsonl_path)
        
        # Print results
        print(f"\n✓ Pipeline completed successfully!")
        print(f"Processed {len(result.nodes)} nodes and {len(result.edges)} edges")
        
        # Print some details about the nodes
        print("\nNodes:")
        for i, node in enumerate(result.nodes[:3]):  # Show first 3 nodes
            print(f"  {i+1}. {node.name} ({node.type})")
            if node.embedding is not None:
                print(f"     Embedding shape: {len(node.embedding)}")
        
        # Print some details about the edges
        print("\nEdges:")
        for i, edge in enumerate(result.edges[:3]):  # Show first 3 edges
            print(f"  {i+1}. {edge.source_node} --[{edge.type}]--> {edge.target_node}")
            if edge.embedding is not None:
                print(f"     Embedding shape: {len(edge.embedding)}")
            if edge.concept_meta:
                print(f"     Concept: {edge.concept_meta}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(jsonl_path):
            os.unlink(jsonl_path)
            print(f"Cleaned up temporary file: {jsonl_path}")


def main():
    """Main test function."""
    print("=" * 60)
    print("PIPELINE TEST WITH SAMPLE DATA")
    print("=" * 60)
    
    success = run_pipeline_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
