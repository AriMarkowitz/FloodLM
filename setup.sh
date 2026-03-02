#!/bin/bash
# setup_environment.sh - Setup guide for FloodLM

set -e

echo "=========================================="
echo "FloodLM Environment Setup Guide"
echo "=========================================="

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Step 1: Create Conda Environment${NC}"
echo "================================"
echo ""
echo "Option A: Using environment.yml (Recommended)"
echo "  conda env create -f environment.yml"
echo ""
echo "Option B: Using requirements.txt"
echo "  conda create -n floodlm python=3.10"
echo "  conda activate floodlm"
echo "  pip install -r requirements.txt"
echo ""

echo -e "${BLUE}Step 2: Activate Environment${NC}"
echo "================================"
echo "  conda activate floodlm"
echo ""

echo -e "${BLUE}Step 3: Configure Data Paths${NC}"
echo "================================"
echo "Edit src/data.py and set:"
echo ""
echo "  # Available options: 'Model_1' or 'Model_2'"
echo "  SELECTED_MODEL = 'Model_1'"
echo ""
echo "Expected directory structure:"
echo "  FloodLM/"
echo "  ├── data/"
echo "  │   ├── Model_1/"
echo "  │   │   └── train/"
echo "  │   │       ├── 1d_nodes_static.csv"
echo "  │   │       ├── 2d_nodes_static.csv"
echo "  │   │       ├── event_0/"
echo "  │   │       │   ├── 1d_nodes_dynamic_all.csv"
echo "  │   │       │   └── 2d_nodes_dynamic_all.csv"
echo "  │   │       └── ..."
echo "  │   └── Model_2/"
echo "  │       └── train/"
echo "  └── src/"
echo ""

echo -e "${BLUE}Step 4: Verify Installation${NC}"
echo "================================"
echo "Run the test script:"
echo "  bash test_pipeline.sh"
echo ""

echo -e "${BLUE}Step 5: Usage Example${NC}"
echo "================================"
echo "Training example:"
echo "  python example_usage.py"
echo ""
echo "Interactive notebook:"
echo "  jupyter notebook"
echo ""

echo -e "${BLUE}Step 6: Switching Models${NC}"
echo "================================"
echo "To switch between Model_1 and Model_2:"
echo ""
echo "  1. Edit src/data.py:"
echo "     SELECTED_MODEL = 'Model_2'  # Change this line"
echo ""
echo "  2. Restart your Python kernel or script"
echo ""

echo -e "${GREEN}=========================================="
echo "Setup Guide Complete!"
echo "==========================================${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Install dependencies: ${YELLOW}conda env create -f environment.yml${NC}"
echo -e "  2. Activate: ${YELLOW}conda activate floodlm${NC}"
echo -e "  3. Configure data path in ${YELLOW}src/data.py${NC}"
echo -e "  4. Test: ${YELLOW}bash test_pipeline.sh${NC}"
