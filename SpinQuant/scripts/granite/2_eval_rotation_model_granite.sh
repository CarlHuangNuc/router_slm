# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

Model_id='/prj/qct/aicechina_scratch/jiayhuan/granite-4.0-1B'
output_dir='output/granite'

# Optional: Test prompt for generation (leave empty to skip)
test_prompt='<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please ensure responses are professional, accurate, and safe.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Please list one IBM Research laboratory located in the United States. You should only output its name and location.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>'

# Build optional arguments
test_prompt_arg=""
[ -n "$test_prompt" ] && test_prompt_arg="--test_prompt \"$test_prompt\""

torchrun --nnodes=1 --nproc_per_node=2 models/granite/rotation_model_generation_granite.py \
--input_model $Model_id \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits 4 \
--a_bits 16 \
--k_bits 8 \
--v_bits 8 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--save_no_had_model_path "$output_dir/no_had_model_random_H/" \
$test_prompt_arg
