#!/bin/bash

# PTQ Evaluation Configuration for Qwen3
# Performs post-training quantization and evaluates model performance

# Model Configuration
Model_id='output/granite/no_had_model_random_H'
output_dir='output/granite'

# Optional: LoRA adapter path (leave empty if not using LoRA)
lora_path=""

# The input fp model should be no had_model rather than origin HF model
if [ -z "$lora_path" ]; then
    Model_id='output/granite/no_had_model_random_H'
fi

# Optional: Calibration dataset path (leave empty to use default wikitext)
calibration_dataset='/prj/qct/aicechina_scratch/jiayhuan/granite/example1/output_qwen3_en.jsonl'

# Optional: Test prompt for generation (leave empty to skip)
test_prompt="<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please ensure responses are professional, accurate, and safe.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Please list one IBM Research laboratory located in the United States. You should only output its name and location.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"

# Build optional arguments
cmd_args=()
[ -n "$lora_path" ] && cmd_args+=(--lora_path "$lora_path")
[ -n "$calibration_dataset" ] && cmd_args+=(--calibration_data "$calibration_dataset")
[ -n "$test_prompt" ] && cmd_args+=(--test_prompt "$test_prompt")

torchrun --nnodes=1 --nproc_per_node=2 models/granite/ptq_granite.py \
--input_model "$Model_id" \
--model_max_length 1024 \
--bf16 True \
--save_safetensors True \
--w_bits 4 \
--w_clip \
--save_qmodel_path "$output_dir/gptq.pth" \
"${cmd_args[@]}"
