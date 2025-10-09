#!/bin/bash
# filepath: /Users/m/Downloads/AISafetyIntervention_LiteratureExtraction/delete_embeddings.sh

# Check if a directory path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

MAIN_DIR="$1"

# Find and delete all directories named 'embeddings' recursively
find "$MAIN_DIR" -type d -name "embeddings" -exec rm -rf {} +

echo "All 'embeddings' folders have been deleted."