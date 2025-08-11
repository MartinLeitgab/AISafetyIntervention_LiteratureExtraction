# Model Provider Package

This package provides a unified interface for AI model providers, with a focus on OpenAI's GPT models and comprehensive batch processing support.

## Architecture

The package follows an abstract base class pattern with the following components:

- **`ModelProvider`**: Abstract base class defining the interface
- **`OpenAIModelProvider`**: Full implementation for OpenAI's GPT models with batch API support

## Features

### Comprehensive OpenAI Integration
- **Single Inference**: Standard chat completions API
- **Concurrent Batch Processing**: Native OpenAI Batch API with concurrent submission and collection
- **Parallel Execution**: Multiple batch jobs submitted simultaneously for maximum throughput

### Batch API Features
- **Concurrent Submission**: All batch chunks submitted simultaneously
- **Parallel Collection**: Results collected from multiple batch jobs concurrently
- **JSONL File Generation**: Automatic creation of batch request files
- **File Upload & Management**: Handles file uploads and cleanup
- **Status Monitoring**: Polls multiple batch job statuses concurrently
- **Result Parsing**: Maintains prompt order across multiple concurrent batches
- **Error Handling**: Comprehensive error handling for all batch operations

### Configuration Options
- **Custom Models**: Support for any OpenAI model (GPT-4, GPT-4o-mini, etc.)
- **Temperature Control**: Configurable response randomness
- **Batch Size Limits**: Configurable batch size limits (default: 50,000)
- **Timeout Settings**: Configurable wait times for batch completion

## Installation

Install the required dependencies:

```bash
pip install openai asyncio
```

## Usage

### Basic Usage

```python
import asyncio
import os
from src.model_provider.openai_provider import OpenAIModelProvider

async def main():
    # Initialize provider
    provider = OpenAIModelProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini",
        temperature=0.2
    )
    
    # Single inference
    response = await provider.infer("What is artificial intelligence?")
    print(response)
    
    # Small batch (uses concurrent requests)
    small_prompts = ["What is AI?", "What is ML?", "What is DL?"]
    responses = await provider.batch_infer(small_prompts)
    
    # Large batch (uses OpenAI Batch API)
    large_prompts = ["Prompt {}".format(i) for i in range(150)]
    batch_responses = await provider.batch_infer(large_prompts)

asyncio.run(main())
```

### Advanced Configuration

```python
provider = OpenAIModelProvider(
    api_key="your-api-key",
    model_name="gpt-4",
    temperature=0.1,
    batch_size_limit=25000  # Custom batch size limit
)
```

### Environment Variables

Set up your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-key"
```

## Batch Processing Details

### All Batch Sizes
- Uses OpenAI's native Batch API exclusively with concurrent processing
- Maximum throughput through parallel batch job submission
- Multiple batch jobs run concurrently, reducing overall completion time
- Automatic concurrent status polling for all batch jobs
- Handles batches up to 50,000 prompts per chunk

### Error Handling
- **API Errors**: Captures and reports OpenAI API errors
- **Network Issues**: Handles connection problems with fallback
- **Parsing Errors**: Graceful handling of malformed responses
- **Timeout Handling**: Configurable timeouts for batch operations
- **Order Preservation**: Maintains original prompt order even with errors

## Integration with SOAR-5

This model provider is specifically designed for the SOAR-5 project's AI safety research pipeline:

1. **Research Paper Processing**: Efficient batch processing of multiple papers
2. **Cost Optimization**: Uses batch API for large-scale processing
3. **Reliability**: Robust error handling for production research workflows
4. **Monitoring**: Built-in status monitoring for long-running batch jobs

## Implementation Details

### Key Methods
- `infer(prompt: str) -> str`: Single inference
- `batch_infer(prompts: List[str]) -> List[str]`: Concurrent batch processing using OpenAI Batch API
- `_submit_all_batches()`: Concurrent submission of all batch chunks
- `_collect_all_results()`: Concurrent collection of results from all batch jobs
- `_wait_for_batch_completion()`: Status monitoring for individual batch jobs
- `_parse_batch_results()`: Result parsing with order preservation

### File Management
- Uses temporary files for batch processing
- Automatic cleanup of temporary files
- Proper file handle management

### Error Recovery
- Graceful handling of partial failures within batches
- Detailed error reporting for individual prompts
- Maintains result order even with failures

## Files

- `model_provider/model_provider.py`: Abstract base class
- `model_provider/openai_provider.py`: Complete OpenAI implementation
- `example_usage.py`: Comprehensive usage examples with different batch sizes
- `requirements.txt`: Package dependencies

## Testing

Run the example script to test the implementation:

```bash
export OPENAI_API_KEY="your-api-key"
python src/example_usage.py
```

The example includes:
- Single inference demonstration
- Small batch processing (Batch API)
- Large batch processing (Batch API) - commented out for safety
- Error handling demonstration

## Cost Considerations

- **Batch API**: 50% cost reduction for large batches
- **Model Selection**: Use `gpt-4o-mini` for cost-effective testing
- **Batch Size**: Optimize batch sizes based on your use case
- **Temperature**: Lower values (0.1-0.3) for consistent outputs

## Production Recommendations

1. **Monitor Batch Jobs**: Large batches can take hours to complete
2. **Set Reasonable Timeouts**: Default 1-hour timeout for batch completion
3. **Handle Partial Failures**: Individual prompts may fail while others succeed
4. **Use Environment Variables**: Never hardcode API keys
5. **Log Operations**: Add logging for production monitoring
