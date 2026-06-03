#!/bin/bash

# PTQ Evaluation Configuration for Qwen3
# Performs post-training quantization and evaluates model performance

# Model Configuration
Model_id='output/hunyuan/no_had_model_random_H'
output_dir='output/hunyuan'

# Optional: LoRA adapter path (leave empty if not using LoRA)
lora_path=""

# The input fp model should be no had_model rather than origin HF model
if [ -z "$lora_path" ]; then
    Model_id='output/hunyuan/no_had_model_random_H'
fi

# Optional: Calibration dataset path (leave empty to use default wikitext)
calibration_dataset=''

# Optional: Test prompt for generation (leave empty to skip)
test_prompt="<｜hy_begin▁of▁sentence｜><｜hy_User｜>中国首都在哪里？<｜hy_Assistant｜>"

# Build optional arguments
cmd_args=()
[ -n "$lora_path" ] && cmd_args+=(--lora_path "$lora_path")
[ -n "$calibration_dataset" ] && cmd_args+=(--calibration_data "$calibration_dataset")
[ -n "$test_prompt" ] && cmd_args+=(--test_prompt "$test_prompt")

torchrun --nnodes=1 --nproc_per_node=2 models/hunyuan/ptq_hunyuan.py \
--input_model "$Model_id" \
--model_max_length 1024 \
--bf16 True \
--save_safetensors True \
--w_bits 4 \
--w_clip \
--save_qmodel_path "$output_dir/gptq.pth" \
"${cmd_args[@]}"
