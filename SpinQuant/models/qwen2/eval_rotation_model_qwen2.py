# coding=utf-8

import datetime
import os
from logging import Logger
import contextlib

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen2TokenizerFast, Trainer, default_data_collator
import transformers
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_qwen2_quant import Qwen2ForCausalLM as Qwen2ForCausalLMQuant
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import random_hadamard_matrix
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict

log: Logger = get_logger("spinquant")
from utils import data_utils, eval_utils

from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils
from train_utils import apply_r3_r4, rtn_utils

def prepare_model_float(args, model):
    transformers.set_seed(args.seed)
    model.eval()

    # Rotate the weights
    fuse_norm_utils.fuse_layer_norms(model)
    # apply_r3_r4.rotate_model(model, args)
    # utils.cleanup_memory(verbos=True)

    # quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
    # qlayers = quant_utils.find_qlayers(model)
    # for name in qlayers:
    #     if "down_proj" in name:
    #         had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
    #         qlayers[name].online_full_had = True
    #         qlayers[name].had_K = had_K
    #         qlayers[name].K = K
    #         qlayers[name].fp32_had = args.fp32_had

    return model


class RotateModule(nn.Module):
    def __init__(self, R_init):
        super(RotateModule, self).__init__()
        if R_init is not None:
            self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda")))
        else:
            self.weight = None

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x

def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )

    # Qwen2 v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = Qwen2ForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model = prepare_model_float(ptq_args, model)
    model.seqlen = training_args.model_max_length
    for param in model.parameters():
        param.requires_grad = False
    R1 = random_hadamard_matrix(model.config.hidden_size, "cpu")
    # R1 = None
    model.R1 = RotateModule(R1)
    for i in range(model.config.num_hidden_layers):
        # Each head dim = 128 for Qwen2 model
        R2 = random_hadamard_matrix(
            model.config.hidden_size // model.config.num_attention_heads, "cuda"
        )
        # R2 = None
        model.model.layers[i].self_attn.R2 = RotateModule(R2)
    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    with torch.no_grad():
    #     prompt = "Hey, are you consciours? "
    #     inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to("cuda")
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n怎么使用筷子？<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, truncation=True, return_tensors="pt").to("cuda")
        model.to("cuda")
        generate_ids_inf = model.generate(inputs.input_ids, do_sample=False, max_length=200)
        print("="*100)
        print(tokenizer.batch_decode(generate_ids_inf, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        print("="*100)

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    dist.barrier()



if __name__ == "__main__":
    train()
