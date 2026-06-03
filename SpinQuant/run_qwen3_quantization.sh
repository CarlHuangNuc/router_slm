#!/bin/bash

# Qwen3 SpinQuant Pipeline
# This script runs the complete quantization workflow for Qwen3 models:
# 1. Optimize rotation matrices
# 2. Evaluate rotation model
# 3. Evaluate PTQ (Post-Training Quantization)

set -e  # Exit on error

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create logs directory if it doesn't exist
mkdir -p logs

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qwen3 SpinQuant Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Optimize rotation matrices
echo -e "${YELLOW}[Step 1/3] Optimizing rotation matrices...${NC}"
echo "Running: bash scripts/qwen3/10_optimize_rotation_qwen3.sh"
if bash scripts/qwen3/10_optimize_rotation_qwen3.sh &> logs/10_optimize_rotation_qwen3.log; then
    echo -e "${GREEN}✓ Step 1 completed successfully${NC}"
    echo "  Log saved to: logs/10_optimize_rotation_qwen3.log"
else
    echo -e "${RED}✗ Step 1 failed. Check logs/10_optimize_rotation_qwen3.log for details${NC}"
    exit 1
fi
echo ""

# Step 2: Evaluate rotation model
echo -e "${YELLOW}[Step 2/3] Evaluating rotation model...${NC}"
echo "Running: bash scripts/qwen3/2_eval_rotation_model_qwen3.sh"
if bash scripts/qwen3/2_eval_rotation_model_qwen3.sh &> logs/2_eval_rotation_model_qwen3.log; then
    echo -e "${GREEN}✓ Step 2 completed successfully${NC}"
    echo "  Log saved to: logs/2_eval_rotation_model_qwen3.log"
else
    echo -e "${RED}✗ Step 2 failed. Check logs/2_eval_rotation_model_qwen3.log for details${NC}"
    exit 1
fi
echo ""

# Step 3: Evaluate PTQ
echo -e "${YELLOW}[Step 3/3] Evaluating PTQ (Post-Training Quantization)...${NC}"
echo "Running: bash scripts/qwen3/2_eval_ptq_qwen3.sh"
if bash scripts/qwen3/2_eval_ptq_qwen3.sh &> logs/2_eval_ptq_qwen3.log; then
    echo -e "${GREEN}✓ Step 3 completed successfully${NC}"
    echo "  Log saved to: logs/2_eval_ptq_qwen3.log"
else
    echo -e "${RED}✗ Step 3 failed. Check logs/2_eval_ptq_qwen3.log for details${NC}"
    exit 1
fi

python models/qwen3/merge_r3.py
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All steps completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  - Rotation matrices optimized"
echo "  - Rotation model evaluated"
echo "  - PTQ model evaluated"
echo ""
echo "All logs are saved in the logs/ directory"
