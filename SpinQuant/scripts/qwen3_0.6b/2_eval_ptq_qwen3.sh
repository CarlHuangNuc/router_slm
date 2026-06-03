#!/bin/bash

# PTQ Evaluation Configuration for Qwen3-0.6B
# Performs post-training quantization and evaluates model performance

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_DISABLED=true

# Model Configuration
Model_id='/dfs/data/Qoder/router_slm/Router/qwen3-0.6b'
output_dir='output/qwen3_0.6b'

# Optional: LoRA adapter path (leave empty if not using LoRA)
lora_path=''

# The input fp model should be no_had_model rather than origin HF model
if [ -z "$lora_path" ]; then
    Model_id='output/qwen3_0.6b/no_had_model_random_H'
fi

# Optional: Calibration dataset path (leave empty to use default wikitext)
calibration_dataset='/dfs/data/Qoder/router_slm/SpinQuant/calibration_data_0.6b.json'

# Optional: Test prompt for generation (leave empty to skip)
test_prompt='<|im_start|>user\n中国首都在哪里？<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

# Build optional arguments
lora_arg=""
calib_arg=""
test_prompt_arg=""
[ -n "$lora_path" ] && lora_arg="--lora_path $lora_path"
[ -n "$calibration_dataset" ] && calib_arg="--calibration_data $calibration_dataset"
[ -n "$test_prompt" ] && test_prompt_arg="--test_prompt $test_prompt"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29508 models/qwen3/ptq_qwen3.py \
--input_model "$Model_id" \
--model_max_length 1024 \
--bf16 True \
--save_safetensors True \
--w_bits 4 \
--w_clip \
--save_qmodel_path "$output_dir/gptq.pth" \
$lora_arg \
$calib_arg \
$test_prompt_arg
