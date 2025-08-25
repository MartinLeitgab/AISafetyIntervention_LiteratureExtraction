# Pipeline System

This directory contains a pipeline system for processing graph data using the Chain of Responsibility pattern.

## Overview

The pipeline system allows you to process graph data through multiple stages, where each stage (Flow) can modify the data and pass it to the next stage. This is implemented using the Chain of Responsibility design pattern.

## Architecture

### Core Components

1. **Pipeline**: Main orchestrator that manages the chain of flows
2. **Flow**: Abstract base class for all processing stages
3. **LocalGraph**: Data structure containing nodes and edges with embeddings
4. **GraphNode/GraphEdge**: Extended versions of Node/Edge with embedding support

### Flow Types

1. **EmbedderFlow**: Loads JSONL data, converts to LocalGraph, and adds embeddings using SentenceTransformers
2. **DatabaseDispatchFlow**: Handles database operations and disk storage

## Usage

### Basic Example

```python
from intervention_graph_creation.src.pipeline import Pipeline, EmbedderFlow, DatabaseDispatchFlow

# Create pipeline
pipeline = Pipeline()

# Create flows
embedder_flow = EmbedderFlow()  # Uses BAAI/bge-large-en-v1.5 by default
db_flow = DatabaseDispatchFlow(storage_dir="output")

# Add flows to pipeline
pipeline.push_node(embedder_flow).push_node(db_flow)

# Process a JSONL file
result = pipeline.process("path/to/output.jsonl")
```

### Custom Flow

You can create custom flows by inheriting from the `Flow` base class:

```python
from intervention_graph_creation.src.pipeline.flows.base import Flow
from intervention_graph_creation.src.pipeline.local_graph import LocalGraph

class CustomFlow(Flow):
    def process(self, local_graph: LocalGraph) -> LocalGraph:
        # Your custom processing logic here
        # Modify local_graph as needed
        
        # Pass to next flow in chain
        return self._call_next(local_graph)
```

## Data Structures

### LocalGraph

The main data container that holds:
- `nodes`: List of GraphNode objects
- `edges`: List of GraphEdge objects

### GraphNode

Extends the base Node with:
- `embedding`: numpy array containing the node's embedding (1024 dimensions for BGE-large-en-v1.5)

### GraphEdge

Extends the base Edge with:
- `embedding`: numpy array containing the edge's embedding (1024 dimensions for BGE-large-en-v1.5)
- `concept_meta`: string containing concept metadata (equivalent to LogicalChain title)

## File Structure

```
pipeline/
├── __init__.py
├── pipeline.py              # Main Pipeline class
├── local_graph.py           # LocalGraph data structures
├── example_usage.py         # Usage examples
├── test_embedding_only.py   # Test script for embedding functionality
├── test_basic_pipeline.py   # Basic pipeline tests
├── README.md               # This file
└── flows/
    ├── __init__.py
    ├── base.py             # Abstract Flow base class
    ├── embedder_flow.py    # EmbedderFlow implementation
    └── database_dispatch_flow.py  # DatabaseDispatchFlow implementation
```

## Dependencies

- `sentence-transformers`: For generating embeddings (BAAI/bge-large-en-v1.5)
- `numpy`: For handling embeddings
- `pydantic`: For data validation
- `falkordb`: For database operations
- `torch`: Required by sentence-transformers

## Configuration

### Embedding Model

The EmbedderFlow uses `BAAI/bge-large-en-v1.5` by default, which:
- Generates 1024-dimensional embeddings
- Is downloaded automatically on first use
- Provides high-quality semantic embeddings
- Runs locally (no API costs)

### Database Configuration

The DatabaseDispatchFlow requires:
- FalkorDB running on localhost:6379
- Graph named "AISafetyIntervention"

## Error Handling

The pipeline includes error handling for:
- Missing files
- Model loading errors
- Database connection issues
- Invalid data formats
- Embedding generation failures (falls back to zero vectors)

## Testing

### Run Basic Tests
```bash
python intervention_graph_creation/src/pipeline/test_basic_pipeline.py
```

### Run Embedding Tests
```bash
python intervention_graph_creation/src/pipeline/test_embedding_only.py
```

## Extending the System

To add new flows:

1. Create a new class inheriting from `Flow`
2. Implement the `process` method
3. Add the flow to your pipeline using `push_node()`

The system is designed to be easily extensible while maintaining the chain-of-responsibility pattern.

## Performance

- **Embedding Generation**: ~1-2 seconds per node/edge on CPU
- **Memory Usage**: ~2GB for BGE-large-en-v1.5 model
- **Storage**: Each embedding is 1024-dimensional float32 array (4KB per embedding)
