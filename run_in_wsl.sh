#!/bin/bash
#
# Complete workflow for running semantic compression in WSL
# This script handles everything from environment setup to running the workflow
#

# Display banner
echo "======================================================"
echo "   Semantic Compression Workflow for WSL Environment   "
echo "======================================================"
echo

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're running in WSL
if [ ! -f /proc/sys/kernel/osrelease ] || ! grep -q Microsoft /proc/sys/kernel/osrelease; then
    echo -e "${YELLOW}Warning: This script is designed for WSL. It doesn't appear you're running in WSL.${NC}"
    echo "You can still continue, but it might not work as expected."
    echo
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Python
echo "Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3 using: sudo apt-get update && sudo apt-get install python3"
    exit 1
fi
python3_version=$(python3 --version)
echo -e "${GREEN}✓${NC} $python3_version is installed"

# Check for .env file
echo -e "\nChecking for .env file..."
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating a template...${NC}"
    echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
    echo "Please edit the .env file with your actual OpenAI API key."
else
    if grep -q "OPENAI_API_KEY=" .env; then
        echo -e "${GREEN}✓${NC} .env file found with API key"
    else
        echo -e "${YELLOW}Warning: No OPENAI_API_KEY found in .env file.${NC}"
        echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
        echo "Please edit the .env file with your actual OpenAI API key."
    fi
fi

# Check for virtual environment
echo -e "\nChecking for virtual environment..."
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC} Found existing virtual environment"
    echo "Activating virtual environment..."
    source .venv/bin/activate
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Failed to activate virtual environment.${NC}"
        echo "Continuing without virtual environment."
    else
        echo -e "${GREEN}✓${NC} Virtual environment activated"
    fi
else
    echo "Creating new virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Failed to create virtual environment.${NC}"
        echo "Continuing without virtual environment."
    else
        source .venv/bin/activate
        echo -e "${GREEN}✓${NC} Virtual environment created and activated"
    fi
fi

# Install key packages
echo -e "\nInstalling critical packages..."
python3 -m pip install python-dotenv
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Failed to install python-dotenv.${NC}"
else
    echo -e "${GREEN}✓${NC} python-dotenv installed"
fi

# Install remaining dependencies
echo -e "\nWould you like to install all remaining dependencies? This may take a few minutes."
read -p "Install all dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing dependencies..."
    python3 -m pip install openai numpy tqdm falkordb sentence-transformers
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Some dependencies may not have installed correctly.${NC}"
        echo "Continuing anyway..."
    else
        echo -e "${GREEN}✓${NC} All dependencies installed"
    fi
else
    echo "Skipping dependency installation."
fi

# Set OpenAI API key in environment for this session
echo -e "\nLoading OpenAI API key from .env file..."
if [ -f .env ]; then
    # Export API key from .env file to environment
    export $(grep -v '^#' .env | xargs)
    if [ -n "$OPENAI_API_KEY" ]; then
        echo -e "${GREEN}✓${NC} API key loaded into environment"
    else
        echo -e "${YELLOW}Warning: Could not load API key from .env file${NC}"
    fi
fi

# Check for FalkorDB
echo -e "\nChecking FalkorDB connection..."
# Simple check if the port is open
if command -v nc &> /dev/null; then
    nc -z localhost 6379 &> /dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} FalkorDB port is open (6379)"
    else
        echo -e "${YELLOW}Warning: FalkorDB port 6379 does not appear to be open${NC}"
        echo "Make sure FalkorDB is running before continuing"
    fi
else
    echo "Cannot check FalkorDB connection (nc tool not available)"
fi

# Ask for workflow parameters
echo -e "\n=== Semantic Compression Parameters ==="
read -p "Similarity threshold (0.0-1.0) [default: 0.85]: " threshold
threshold=${threshold:-0.85}

read -p "Maximum cluster size [default: 5]: " cluster_size
cluster_size=${cluster_size:-5}

read -p "Apply merges to database? (y/n) [default: n]: " apply_merges
if [[ $apply_merges =~ ^[Yy]$ ]]; then
    apply_merges="--apply-merges"
else
    apply_merges=""
fi

# Run the workflow
echo -e "\n=== Running Semantic Compression Workflow ==="
echo "Parameters:"
echo "- Threshold: $threshold"
echo "- Cluster size: $cluster_size"
echo "- Apply merges: ${apply_merges:-No}"
echo

# Create direct command to run
cmd="python3 direct_run_semantic_compression.py --threshold=$threshold --cluster-size=$cluster_size $apply_merges"
echo "Running: $cmd"
echo
eval $cmd

# Display completion message
echo -e "\n======================================================"
echo "             Workflow Run Complete                     "
echo "======================================================"
