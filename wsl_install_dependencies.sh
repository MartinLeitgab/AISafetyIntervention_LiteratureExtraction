#!/bin/bash
#
# Helper script for installing semantic compression dependencies in WSL
#

# Display banner
echo "======================================================"
echo "      Installing Semantic Compression Dependencies     "
echo "======================================================"
echo

# Check if we're running in WSL
if [ ! -f /proc/sys/kernel/osrelease ] || ! grep -q Microsoft /proc/sys/kernel/osrelease; then
    echo "This script is designed for WSL. It doesn't appear you're running in WSL."
    echo "You can still continue, but it might not work as expected."
    echo
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is installed
if ! python3 -m pip --version &> /dev/null; then
    echo "Pip is not installed. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
fi

# Check for virtual environment
if [ -d ".venv" ]; then
    echo "Found existing virtual environment. Activating..."
    source .venv/bin/activate
    if [ $? -ne 0 ]; then
        echo "Failed to activate virtual environment. Trying to continue anyway..."
    else
        echo "Virtual environment activated."
    fi
else
    echo "No virtual environment found. Creating new one..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Trying to continue anyway..."
    else
        source .venv/bin/activate
        echo "Virtual environment created and activated."
    fi
fi

# Install required packages
echo
echo "Installing required packages..."
echo

# Required packages
packages=(
    "openai>=1.0.0"
    "python-dotenv>=1.0.0"
    "numpy>=1.20.0"
    "tqdm>=4.64.0"
    "falkordb>=1.0.0"
    "sentence-transformers>=2.2.0"
)

# Install packages one by one to better isolate any failures
for package in "${packages[@]}"; do
    echo "Installing $package..."
    python3 -m pip install $package
    if [ $? -ne 0 ]; then
        echo "Failed to install $package. Continuing with remaining packages..."
    else
        echo "Successfully installed $package."
    fi
    echo
done

echo "Installation process completed."
echo
echo "To verify the setup, run:"
echo "python3 setup_semantic_compression.py"
echo
echo "To run the workflow, run:"
echo "python3 run_semantic_compression_workflow.py"
