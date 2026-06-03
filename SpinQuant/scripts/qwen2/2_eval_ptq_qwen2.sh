#!/bin/bash

# PTQ Evaluation Configuration for Qwen3
# Performs post-training quantization and evaluates model performance

# Model Configuration
Model_id='/prj/qct/aicechina_scratch/ruzhongl/llm/qwen2_5_omini/llm_notebook_ce/Ali/qwen2.5-3b-omni/helper_scripts/llm/sub_model/llm'
output_dir='output/qwen2.5-3b-vl'

# Optional: LoRA adapter path (leave empty if not using LoRA)
lora_path=""

# The input fp model should be no had_model rather than origin HF model
if [ -z "$lora_path" ]; then
    Model_id='output/qwen2.5-3b-vl/no_had_model_random_H'
fi

# Optional: Calibration dataset path (leave empty to use default wikitext)
calibration_dataset='/prj/qct/aicechina_scratch/weihuan/eval/qwen-nb/llm_notebook_ce/qwen2.5-gptq-llm/sys_pipeline/train_data_new.json'

# Optional: Test prompt for generation (leave empty to skip)
test_prompt="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n王伟长得矮胖，同学刘勇给他取了个“武大郎”的外号。刘勇侵犯了王伟的____\n\nA. 名誉权\nB. 姓名权\nC. 荣誉权\nD. 肖像权\n<|im_end|>\n<|im_start|>assistant\n"

# Build optional arguments
cmd_args=()
[ -n "$lora_path" ] && cmd_args+=(--lora_path "$lora_path")
[ -n "$calibration_dataset" ] && cmd_args+=(--calibration_data "$calibration_dataset")
[ -n "$test_prompt" ] && cmd_args+=(--test_prompt "$test_prompt")

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29508 -m models.qwen2.ptq_qwen2 \
--input_model "$Model_id" \
--model_max_length 1024 \
--bf16 True \
--save_safetensors True \
--w_bits 8 \
--w_clip \
--save_qmodel_path "$output_dir/gptq.pth" \
"${cmd_args[@]}"
