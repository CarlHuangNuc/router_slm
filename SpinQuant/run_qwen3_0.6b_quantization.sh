#!/bin/bash

# Qwen3-0.6B SpinQuant Pipeline
# This script runs the complete quantization workflow:
# 1. Optimize rotation matrices
# 2. Apply rotation and generate rotated model
# 3. GPTQ Post-Training Quantization

set -eo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

mkdir -p logs

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qwen3-0.6B SpinQuant Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Step 1: Optimize rotation matrices
echo -e "${YELLOW}[Step 1/3] Optimizing rotation matrices...${NC}"
echo "Running: bash scripts/qwen3_0.6b/10_optimize_rotation_qwen3.sh"
if bash scripts/qwen3_0.6b/10_optimize_rotation_qwen3.sh 2>&1 | tee logs/0.6b_optimize_rotation.log; then
    echo -e "${GREEN}✓ Step 1 completed successfully${NC}"
else
    echo -e "${RED}✗ Step 1 failed. Check logs/0.6b_optimize_rotation.log${NC}"
    exit 1
fi
echo ""

# Step 2: Apply rotation and generate rotated model
echo -e "${YELLOW}[Step 2/3] Generating rotated model...${NC}"
echo "Running: bash scripts/qwen3_0.6b/2_eval_rotation_model_qwen3.sh"
if bash scripts/qwen3_0.6b/2_eval_rotation_model_qwen3.sh 2>&1 | tee logs/0.6b_rotation_model.log; then
    echo -e "${GREEN}✓ Step 2 completed successfully${NC}"
else
    echo -e "${RED}✗ Step 2 failed. Check logs/0.6b_rotation_model.log${NC}"
    exit 1
fi
echo ""

# Step 3: GPTQ PTQ quantization
echo -e "${YELLOW}[Step 3/3] Running GPTQ Post-Training Quantization...${NC}"
echo "Running: bash scripts/qwen3_0.6b/2_eval_ptq_qwen3.sh"
if bash scripts/qwen3_0.6b/2_eval_ptq_qwen3.sh 2>&1 | tee logs/0.6b_ptq.log; then
    echo -e "${GREEN}✓ Step 3 completed successfully${NC}"
else
    echo -e "${RED}✗ Step 3 failed. Check logs/0.6b_ptq.log${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All steps completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output:"
echo "  - Rotation matrix: output/qwen3_0.6b/rotation/R.bin"
echo "  - Rotated model:   output/qwen3_0.6b/no_had_model_random_H/"
echo "  - Quantized model: output/qwen3_0.6b/gptq.pth"
echo ""
echo "All logs saved in logs/ directory"
