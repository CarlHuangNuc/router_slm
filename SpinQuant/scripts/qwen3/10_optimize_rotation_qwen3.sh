#!/bin/bash

# Rotation Optimization Configuration for Qwen3
# This script optimizes rotation matrices for quantization-aware training

# Fix protobuf/tensorboard incompatibility (protobuf >= 4.x vs tensorboard)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_DISABLED=true

# Model Configuration
Model_id='/prj/qct/aicechina_scratch/models/Qwen3-1.7B'
output_dir='output/qwen3_1.7b'

# Calibration Dataset Configuration
# Specify custom calibration dataset path (JSON/JSONL format)
# Leave empty to use default wikitext dataset
# Example: calibration_dataset='/path/to/your/calibration_data.jsonl'
calibration_dataset='/prj/qct/aicechina_scratch/weihuan/eval/qwen3-llm/llm_notebook_ce/Ali/qwen3-1.7b/example1/dataset/qwen3_train_data.json'

# Build optional arguments
calib_arg=""
[ -n "$calibration_dataset" ] && calib_arg="--calibration_data $calibration_dataset"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29505 models/qwen3/optimize_rotation_qwen3.py \
--input_model "$Model_id" \
--output_rotation_path "$output_dir/rotation" \
--output_dir "$output_dir" \
--logging_dir "$output_dir" \
--model_max_length 1024 \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
--w_bits 4 \
--a_bits 16 \
--k_bits 16 \
--v_bits 16 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
$calib_arg
