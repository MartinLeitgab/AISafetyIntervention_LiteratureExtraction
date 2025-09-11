# Running Semantic Compression in WSL

This guide explains how to run the semantic compression workflow in WSL (Windows Subsystem for Linux).

## Overview

Running the workflow in WSL may encounter some issues with Python package installation and virtual environments. The scripts in this guide are designed to help you overcome these challenges.

## Prerequisites

- WSL installed on your Windows system
- Python 3.7+ installed in WSL
- FalkorDB running (either in WSL or Windows)
- OpenAI API key

## Quick Start

For the easiest experience, use the all-in-one WSL script:

```bash
# From your WSL terminal
cd /mnt/c/Users/Mitali\ Raj/Downloads/Github/AISafetyIntervention_LiteratureExtraction
chmod +x run_in_wsl.sh
./run_in_wsl.sh
```

This script will:
1. Check your environment
2. Install necessary packages
3. Set up your OpenAI API key
4. Run the workflow with your specified parameters

## Detailed Instructions

### 1. Edit your .env file

Make sure your `.env` file contains your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key
```

### 2. Install dependencies

If you're having trouble with package installation, use the WSL-specific installation script:

```bash
chmod +x wsl_install_dependencies.sh
./wsl_install_dependencies.sh
```

### 3. Run the workflow directly

To bypass environment checks and run the workflow directly:

```bash
python3 direct_run_semantic_compression.py --threshold=0.85 --cluster-size=5
```

Add `--apply-merges` if you want to apply the suggested merges to the database.

## Troubleshooting

### Package Installation Issues

If you're having trouble installing packages in WSL:

1. Try installing them one by one:
   ```bash
   python3 -m pip install python-dotenv
   python3 -m pip install openai
   python3 -m pip install falkordb
   ```

2. If sentence-transformers is causing issues, try:
   ```bash
   python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python3 -m pip install sentence-transformers
   ```

### FalkorDB Connection Issues

Make sure FalkorDB is running and accessible on port 6379. You can check with:

```bash
nc -z localhost 6379
```

If running FalkorDB in Windows, ensure WSL can access Windows ports.

### OpenAI API Key Issues

If your API key isn't being recognized:

1. Check that it's properly formatted in the `.env` file
2. Export it directly in your terminal:
   ```bash
   export OPENAI_API_KEY=sk-your-actual-api-key
   ```

## Files

- `run_in_wsl.sh`: All-in-one script for WSL users
- `wsl_install_dependencies.sh`: Helper script for installing dependencies in WSL
- `direct_run_semantic_compression.py`: Python script that runs the workflow directly
- `.env`: File containing your OpenAI API key
