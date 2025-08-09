#!/bin/bash
# Setup script for creating conda environments for all DNA models

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "DNA Model Environment Setup"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Found conda at: $(which conda)${NC}"
conda --version

# Parse command line arguments
FORCE=false
VERIFY=false
SPECIFIC_ENV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --env)
            SPECIFIC_ENV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force] [--verify] [--env ENV_NAME]"
            exit 1
            ;;
    esac
done

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Run the Python setup script
if [ "$VERIFY" = true ]; then
    echo -e "${YELLOW}Verifying environments...${NC}"
    python scripts/setup_environments.py --verify
elif [ -n "$SPECIFIC_ENV" ]; then
    echo -e "${YELLOW}Setting up environment: $SPECIFIC_ENV${NC}"
    if [ "$FORCE" = true ]; then
        python scripts/setup_environments.py --env "$SPECIFIC_ENV" --force
    else
        python scripts/setup_environments.py --env "$SPECIFIC_ENV"
    fi
else
    echo -e "${YELLOW}Setting up all environments...${NC}"
    if [ "$FORCE" = true ]; then
        python scripts/setup_environments.py --force
    else
        python scripts/setup_environments.py
    fi
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Setup completed successfully!${NC}"
    
    # Show available environments
    echo ""
    echo "Available environments:"
    conda env list | grep -E "(dnabert|nucleotide|prokbert|grover|gena|inherit|hyena|evo|caduceus)"
else
    echo -e "${RED}Setup failed. Please check the logs above.${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Activate an environment: conda activate <env_name>"
echo "2. Run tests: python -m pytest tests/"
echo "3. Run benchmarks: python scripts/run_benchmark.py"
echo "=========================================="