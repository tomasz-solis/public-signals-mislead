#!/bin/bash

# Complete analysis runner for public-signals-mislead project
# Usage: ./run_analysis.sh

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "================================================================================"
echo "  Public Signals Mislead: Complete Analysis"
echo "================================================================================"
echo ""

# Resolve Python command
BOOTSTRAP_PYTHON=""
if [ -x "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="python"
    PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="python3"
    PYTHON_CMD="python3"
else
    echo -e "${YELLOW}No Python interpreter found on PATH.${NC}"
    exit 1
fi

if [ ! -x "venv/bin/python" ] && [ ! -x ".venv/bin/python" ]; then
    echo -e "${BLUE}Creating local virtual environment...${NC}"
    $BOOTSTRAP_PYTHON -m venv venv
    PYTHON_CMD="venv/bin/python"
    echo ""
fi

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python_version=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"
echo ""

# Check project install
echo -e "${BLUE}Checking project install...${NC}"
if ! $PYTHON_CMD -c "import src, config, pandas, scipy, plotly" 2>/dev/null; then
    echo -e "${BLUE}Installing project in editable mode...${NC}"
    $PYTHON_CMD -m pip install -e .
    echo ""
fi

echo -e "${GREEN}✅ Project environment ready${NC}"
echo ""

# Step 1: Apply decision context
echo "================================================================================"
echo "  Step 1/3: Applying Decision Context"
echo "================================================================================"
echo ""
$PYTHON_CMD scripts/apply_outcomes.py
echo ""

# Step 2: Statistical analysis
echo "================================================================================"
echo "  Step 2/3: Running Decision-Support Analysis"
echo "================================================================================"
echo ""
$PYTHON_CMD src/analysis/statistical_analysis.py
echo ""

# Step 3: Generate visualizations
echo "================================================================================"
echo "  Step 3/3: Generating Interactive Charts"
echo "================================================================================"
echo ""
$PYTHON_CMD scripts/generate_visualizations.py
echo ""

# Summary
echo "================================================================================"
echo -e "  ${GREEN}✅ ANALYSIS COMPLETE${NC}"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  📊 Statistical results: data/validation/statistical_results.csv"
echo "  📈 Interactive charts:  results/figures/*.html"
echo ""
echo "View main chart:"
echo "  open results/figures/decay_vs_action.html"
echo ""
echo "Or browse all charts:"
echo "  open results/figures/"
echo ""
