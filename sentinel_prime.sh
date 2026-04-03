#!/bin/bash

# Sentinel Prime: End-to-End Bootstrap Script (Phase 22)
# This script handles environment calibration, dependency auditing, 
# and provides a unified entry point for the Proctoring platform.

# Color palette for Sentinel UI
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${PURPLE}===================================================${NC}"
echo -e "${PURPLE}      PROJECT SENTINEL: AI PROCTORING PLATFORM     ${NC}"
echo -e "${PURPLE}               PHASE: SENTINEL PRIME               ${NC}"
echo -e "${PURPLE}===================================================${NC}"

# --- CALIBRATION: Path Management ---
# Automatically set PYTHONPATH to include the root directory
# This resolves "ModuleNotFoundError: No module named 'ml'" and other path issues.
export PROJ_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJ_ROOT:$PROJ_ROOT/legacy_system
echo -e "${BLUE}[1/3] Calibrating Environment Paths...${NC}"
echo -e "      PYTHONPATH: $PYTHONPATH"

# --- AUDIT: Dependency Health Check ---
echo -e "${BLUE}[2/3] Auditing Dependency Health...${NC}"

check_py_module() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "      [${GREEN}OK${NC}] $1"
    else
        echo -e "      [${RED}MISSING${NC}] $1 - Attempting auto-fix..."
        pip install $1
    fi
}

check_py_module "mediapipe"
check_py_module "cv2"
check_py_module "torch"
check_py_module "yaml"
check_py_module "numpy"

# --- EXECUTION: Unified Entry Point ---
echo -e "${BLUE}[3/3] System Ready.${NC}"
echo -e "      Select target node to launch:"
echo -e "      1) ${CYAN}Sentinel Backend${NC} (Django API + ML Engine)"
echo -e "      2) ${CYAN}Sentinel Frontend${NC} (React Overwatch & Arena)"
echo -e "      3) ${CYAN}Sentinel Health Check${NC} (Run Full Diagnostics)"
echo -e "      q) Quit"

read -p "Selection: " choice

case $choice in
    1)
        echo -e "${GREEN}Launching Sentinel Backend...${NC}"
        cd legacy_system && python3 manage.py runserver
        ;;
    2)
        echo -e "${GREEN}Launching Sentinel Frontend (Vite)...${NC}"
        cd react_proctoring_app && npm run dev
        ;;
    3)
        echo -e "${GREEN}Running Sentinel Diagnostics...${NC}"
        python3 -c "from proctoring_ml_module.api.inference_interface import create_engine; engine = create_engine(); print('SUCCESS: 7-Signal Engine Initialized Correctly')"
        ;;
    q)
        echo "Exiting Sentinel Prime."
        exit 0
        ;;
    *)
        echo "Invalid selection."
        ;;
esac
