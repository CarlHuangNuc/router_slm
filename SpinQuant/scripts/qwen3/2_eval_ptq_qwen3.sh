#!/bin/bash

# PTQ Evaluation Configuration for Qwen3
# Performs post-training quantization and evaluates model performance

# Fix protobuf/tensorboard incompatibility (protobuf >= 4.x vs tensorboard)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_DISABLED=true

# Model Configuration
Model_id='/prj/qct/aicechina_scratch/models/Qwen3-1.7B'
output_dir='output/qwen3_1.7b'

# Optional: LoRA adapter path (leave empty if not using LoRA)
lora_path=''

# The input fp model should be no had_model rather than origin HF model
if [ -z "$lora_path" ]; then
    Model_id='output/qwen3_1.7b/no_had_model_random_H'
fi

# Optional: Calibration dataset path (leave empty to use default wikitext)
calibration_dataset='/prj/qct/aicechina_scratch/weihuan/eval/qwen3-llm/llm_notebook_ce/Ali/qwen3-1.7b/example1/dataset/qwen3_train_data.json'

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
