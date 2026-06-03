# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

Model_id='/prj/qct/aicechina_scratch/ruzhongl/llm/qwen2_5_omini/llm_notebook_ce/Ali/qwen2.5-3b-omni/helper_scripts/llm/sub_model/llm'
output_dir='/prj/qct/aicechina_scratch/weihuan/eval/SpinQuant/output/qwen2.5-3b-vl'

PYTHONPATH=. torchrun --nnodes=1 --nproc_per_node=1 models/qwen2/optimize_rotation_qwen2.py \
--input_model $Model_id  \
--output_rotation_path $output_dir/rotation \
--output_dir $output_dir \
--logging_dir $output_dir \
--model_max_length 2048 \
--fp16 False \
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
--report_to none \
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
