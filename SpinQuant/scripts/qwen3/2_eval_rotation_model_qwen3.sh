#!/bin/bash

# Rotation Model Generation and Evaluation for Qwen3
# Applies rotation matrices and evaluates the rotated model

# Fix protobuf/tensorboard incompatibility (protobuf >= 4.x vs tensorboard)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_DISABLED=true

# Model Configuration
Model_id='/prj/qct/aicechina_scratch/models/Qwen3-1.7B'
output_dir='output/qwen3_1.7b'

# Optional: Test prompt for generation (leave empty to skip)
test_prompt='<|im_start|>user\n中国首都在哪里？<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

# Build optional arguments
test_prompt_arg=""
[ -n "$test_prompt" ] && test_prompt_arg="--test_prompt \"$test_prompt\""

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29505 models/qwen3/rotation_model_generation_qwen3.py \
--input_model "$Model_id" \
--model_max_length 1024 \
--bf16 True \
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
--rotate \
--save_no_had_model_path "$output_dir/no_had_model_random_H/" \
--optimized_rotation_path "$output_dir/rotation/R.bin" \
$test_prompt_arg
